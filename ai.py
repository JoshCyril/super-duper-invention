# app.py - Streamlit Markdown Knowledge Base Application

import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from pathlib import Path
import yaml
import random
import urllib.parse
import shutil
from streamlit_file_browser import st_file_browser  # Custom file explorer component0
from streamlit_ace import st_ace  # Ace editor component1
import re

# Initialize vault directory
VAULT_PATH = Path("vault")
VAULT_PATH.mkdir(exist_ok=True)
CONFIG_PATH = VAULT_PATH / "config.yml"

# Page config
st.set_page_config(page_title="Knowledge Base", layout="wide", initial_sidebar_state="expanded")

# Session state initialization
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False

# Custom CSS for wiki-links and graph container
st.markdown("""
<style>
.wiki-link {
    background-color: #e6f7ff;
    color: #1a73e8;
    padding: 2px 4px;
    border-radius: 4px;
    text-decoration: none;
}
.graph-container {
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)

def load_folder_colors():
    """Load folder colors from config.yml"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
            return config.get("folder_colors", {})
    return {}

def save_folder_colors(colors):
    """Save folder colors to config.yml"""
    config = {"folder_colors": colors}
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def highlight_wiki_links(text):
    """Wrap wiki-links [[Link]] in clickable anchors"""
    return re.sub(
        r'\[\[(.*?)\]\]',
        r'<a href="?node=\1" class="wiki-link">\1</a>',
        text
    )

def parse_vault_yaml(root_path, data):
    """Recursively create folders and markdown files from YAML definition"""
    for key, value in data.items():
        if key == "folder_colors":
            continue
        if isinstance(value, dict) and "content" in value:
            # It's a note
            file_path = root_path / f"{key}.md"
            content = value.get("content", "")
            file_path.write_text(content, encoding="utf-8")
        else:
            # Subfolder
            new_folder = root_path / key
            new_folder.mkdir(exist_ok=True)
            if isinstance(value, dict):
                parse_vault_yaml(new_folder, value)

# Load existing folder colors
folder_colors = load_folder_colors()

# --- Sidebar: Vault options and file browser ---
with st.sidebar:
    st.header("Vault")
    mode = st.radio("Select Vault Source:", ["Local Vault", "Upload vault.yml"], index=0)
    if mode == "Upload vault.yml":
        uploaded = st.file_uploader("Upload vault.yml", type=['yaml', 'yml'])
        if uploaded:
            try:
                data = yaml.safe_load(uploaded.read().decode("utf-8"))
                if isinstance(data, dict):
                    # Clear current vault
                    shutil.rmtree(VAULT_PATH)
                    VAULT_PATH.mkdir(exist_ok=True)
                    # If folder_colors provided, save to config
                    if "folder_colors" in data:
                        new_colors = data["folder_colors"]
                        save_folder_colors(new_colors)
                        folder_colors = new_colors
                    # Parse vault structure
                    parse_vault_yaml(VAULT_PATH, data)
                    st.success("Vault imported successfully.")
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to load vault.yml: {e}")

    st.markdown("### 📁 File Explorer")
    # Use streamlit-file-browser to display vault
    event = st_file_browser(
        str(VAULT_PATH),
        key="file_browser",
        extentions=['.md'],
        show_preview=False,
        show_choose_file=False,
        show_download_file=False,
        show_new_folder=True,
        show_upload_file=True
    )
    if event and event.get("path"):
        selected_path = Path(event["path"])
        if selected_path.is_file():
            rel_path = str(selected_path.relative_to(VAULT_PATH))
            st.session_state.selected_file = rel_path
            st.session_state.edit_mode = False
            st.experimental_rerun()

    st.markdown("---")
    with st.form("new_file_form"):
        new_file_name = st.text_input("New File Name")
        if st.form_submit_button("Create File") and new_file_name:
            new_path = VAULT_PATH / f"{new_file_name}.md"
            new_path.touch(exist_ok=True)
            st.session_state.selected_file = str(new_path.relative_to(VAULT_PATH))
            st.session_state.edit_mode = True
            st.experimental_rerun()

# Handle node links from URL query param
query_params = st.experimental_get_query_params()
if "node" in query_params:
    node_path = urllib.parse.unquote(query_params["node"][0])
    # Append .md if not present
    node_path = node_path if node_path.lower().endswith('.md') else f"{node_path}.md"
    if (VAULT_PATH / node_path).exists():
        st.session_state.selected_file = node_path
        st.session_state.edit_mode = False
        st.experimental_rerun()

# Main content area
if st.session_state.selected_file:
    file_path = VAULT_PATH / st.session_state.selected_file
    col1, col2 = st.columns([3, 2])

    with col1:
        if st.session_state.edit_mode:
            # Edit mode with ACE editor
            content = file_path.read_text(encoding="utf-8")
            edited = st_ace(
                value=content,
                language='markdown',
                theme='github',
                key='editor',
                height=400,
                wrap=True
            )
            if st.button("Save Changes"):
                file_path.write_text(edited, encoding="utf-8")
                st.success("Saved!")
                st.session_state.edit_mode = False
                st.experimental_rerun()
            if st.button("Cancel"):
                st.session_state.edit_mode = False
                st.experimental_rerun()
        else:
            # Read-only view mode
            st.markdown(f"### {file_path.stem}")
            content = file_path.read_text(encoding="utf-8")
            highlighted = highlight_wiki_links(content)
            st.markdown(highlighted, unsafe_allow_html=True)
            if st.button("Edit File"):
                st.session_state.edit_mode = True
                st.experimental_rerun()

    with col2:
        # Build graph of markdown notes
        def extract_links(text):
            return re.findall(r'\[\[(.*?)\]\]', text)

        def build_graph(vault_path, selected_node=None):
            G = nx.Graph()
            files = list(Path(vault_path).rglob("*.md"))

            global folder_colors
            # Ensure color for each folder
            folders = set(str(f.relative_to(vault_path).parent) for f in files)
            for folder in folders:
                if folder not in folder_colors:
                    folder_colors[folder] = f"#{random.randint(0, 0xFFFFFF):06x}"
            save_folder_colors(folder_colors)

            for file in files:
                stem = file.stem
                rel_path = str(file.relative_to(vault_path))
                folder = str(file.relative_to(vault_path).parent)
                color = folder_colors.get(folder, "#cccccc")
                node_attrs = {"title": rel_path, "color": color}
                if selected_node and stem == selected_node:
                    node_attrs.update({"borderWidth": 4, "borderColor": "#ff0000"})
                G.add_node(stem, **node_attrs)

            for file in files:
                src = file.stem
                content = file.read_text(encoding="utf-8")
                links = extract_links(content)
                for link in links:
                    if link in G:
                        G.add_edge(src, link)
            return G

        try:
            selected_node = Path(st.session_state.selected_file).stem if st.session_state.selected_file else None
            G = build_graph(VAULT_PATH, selected_node)
            net = Network(height='500px', width='100%', notebook=True)
            net.from_nx(G)
            net.save_graph("graph.html")

            # Modify HTML to enable node click navigation
            with open("graph.html", 'r', encoding="utf-8") as f:
                html_content = f.read()
            js_code = """
            <script>
            document.addEventListener("DOMContentLoaded", function() {
                setTimeout(() => {
                    const nodes = document.querySelectorAll(".node");
                    nodes.forEach(node => {
                        node.style.cursor = "pointer";
                        node.addEventListener("click", function() {
                            const title = this.getAttribute("title");
                            window.location.search = "?node=" + encodeURIComponent(title);
                        });
                    });
                }, 1000);
            });
            </script>
            """
            html_content = html_content.replace("</body>", js_code + "</body>")
            html_content = html_content.replace('<body>', '<body style="margin:0; padding:0; border:none;">')
            with open("graph.html", 'w', encoding="utf-8") as f:
                f.write(html_content)
            with open("graph.html", 'r', encoding="utf-8") as f:
                components.html(f.read(), height=600)
        except Exception as e:
            st.warning("Graph generation failed.")
            st.code(str(e))
else:
    st.info("Select a file from the sidebar to begin")
