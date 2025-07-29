import streamlit as st
import os
import re
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from pathlib import Path
import yaml
import random
import urllib.parse
from icecream import ic
import markdown
from markdown.extensions import codehilite, tables, toc
import time

# Initialize vault directory
VAULT_PATH = Path("vault")
VAULT_PATH.mkdir(exist_ok=True)

# YAML config file path
CONFIG_PATH = VAULT_PATH / "config.yml"

# Page config
st.set_page_config(page_title="GraphIQ", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'file_tree_expanded' not in st.session_state:
    st.session_state.file_tree_expanded = {}
if 'graph_update_trigger' not in st.session_state:
    st.session_state.graph_update_trigger = 0

# Custom CSS for VSCode-like interface
st.markdown("""
    <style>
    /* VSCode-like file tree */
    .file-tree {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 13px;
        background-color: #1e1e1e;
        color: #cccccc;
        padding: 8px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    
    .file-item {
        display: flex;
        align-items: center;
        padding: 2px 4px;
        cursor: pointer;
        border-radius: 3px;
        margin: 1px 0;
        transition: background-color 0.1s;
    }
    
    .file-item:hover {
        background-color: #2a2d2e;
    }
    
    .file-item.selected {
        background-color: #094771;
        color: #ffffff;
    }
    
    .file-icon {
        margin-right: 6px;
        font-size: 14px;
    }
    
    .folder-toggle {
        margin-right: 4px;
        font-size: 10px;
        cursor: pointer;
        color: #8c8c8c;
        user-select: none;
    }
    
    .folder-content {
        margin-left: 16px;
        border-left: 1px solid #3c3c3c;
        padding-left: 8px;
    }
    
    /* Wiki links styling */
    .wiki-link {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 2px 6px;
        border-radius: 4px;
        text-decoration: none;
        border: 1px solid #bbdefb;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .wiki-link:hover {
        background-color: #bbdefb;
        text-decoration: none;
    }
    
    /* Enhanced markdown editor */
    .markdown-editor {
        border: 1px solid #ddd;
        border-radius: 8px;
        overflow: hidden;
        background: white;
    }
    
    .editor-toolbar {
        background: #f8f9fa;
        border-bottom: 1px solid #dee2e6;
        padding: 8px 12px;
        display: flex;
        gap: 4px;
        flex-wrap: wrap;
    }
    
    .toolbar-btn {
        background: #fff;
        border: 1px solid #ced4da;
        border-radius: 4px;
        padding: 6px 10px;
        cursor: pointer;
        font-size: 13px;
        transition: all 0.2s;
        color: #495057;
    }
    
    .toolbar-btn:hover {
        background: #e9ecef;
        border-color: #adb5bd;
    }
    
    .toolbar-btn:active {
        background: #dee2e6;
    }
    
    .toolbar-separator {
        width: 1px;
        background: #dee2e6;
        margin: 4px 4px;
    }
    
    /* Editor textarea */
    .stTextArea textarea {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
        border: none !important;
        resize: vertical !important;
    }
    
    /* Graph container */
    .graph-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        overflow: hidden;
        background: white;
    }
    
    /* Status bar */
    .status-bar {
        background: #f8f9fa;
        padding: 4px 12px;
        font-size: 12px;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
    }
    
    /* Markdown preview */
    .markdown-preview {
        padding: 20px;
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        line-height: 1.6;
    }
    
    .markdown-preview h1, .markdown-preview h2, .markdown-preview h3 {
        color: #2c3e50;
        margin-top: 24px;
        margin-bottom: 16px;
    }
    
    .markdown-preview code {
        background: #f8f9fa;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    }
    
    .markdown-preview pre {
        background: #f8f9fa;
        padding: 16px;
        border-radius: 6px;
        overflow-x: auto;
        border-left: 4px solid #007acc;
    }
    
    .markdown-preview blockquote {
        border-left: 4px solid #dfe2e5;
        padding-left: 16px;
        margin-left: 0;
        color: #6a737d;
    }
    </style>
""", unsafe_allow_html=True)

# === YAML CONFIG HANDLING ===
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

# Load folder colors from YAML
folder_colors = load_folder_colors()

def render_vscode_file_tree():
    """Render VSCode-like file tree"""
    st.markdown('<div class="file-tree">', unsafe_allow_html=True)
    
    def render_tree_item(path, level=0):
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        html_content = ""
        
        for item in items:
            rel_path = str(item.relative_to(VAULT_PATH))
            indent = "  " * level
            
            if item.is_dir():
                is_expanded = st.session_state.file_tree_expanded.get(rel_path, False)
                toggle_icon = "▼" if is_expanded else "▶"
                
                html_content += f'''
                <div class="file-item" onclick="toggleFolder('{rel_path}')">
                    {indent}<span class="folder-toggle">{toggle_icon}</span>
                    <span class="file-icon">📁</span>
                    <span>{item.name}</span>
                </div>
                '''
                
                if is_expanded:
                    html_content += f'<div class="folder-content">{render_tree_item(item, level + 1)}</div>'
            else:
                if item.name.lower().endswith('.md'):
                    selected_class = "selected" if st.session_state.selected_file == rel_path else ""
                    html_content += f'''
                    <div class="file-item {selected_class}" onclick="selectFile('{rel_path}')">
                        {indent}<span class="file-icon">📄</span>
                        <span>{item.stem}</span>
                    </div>
                    '''
        
        return html_content
    
    tree_html = render_tree_item(VAULT_PATH)
    st.markdown(tree_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # JavaScript for file tree interaction
    st.markdown("""
    <script>
    function toggleFolder(path) {
        // Send folder toggle to Streamlit
        const event = new CustomEvent('streamlit:folder-toggle', {
            detail: { path: path }
        });
        window.dispatchEvent(event);
    }
    
    function selectFile(path) {
        // Send file selection to Streamlit
        const event = new CustomEvent('streamlit:file-select', {
            detail: { path: path }
        });
        window.dispatchEvent(event);
        
        // Update URL parameters
        const url = new URL(window.location);
        url.searchParams.set('file', path);
        window.history.pushState({}, '', url);
    }
    
    // Listen for custom events
    window.addEventListener('streamlit:folder-toggle', function(e) {
        // This would need to be handled by Streamlit session state
        console.log('Toggle folder:', e.detail.path);
    });
    
    window.addEventListener('streamlit:file-select', function(e) {
        console.log('Select file:', e.detail.path);
    });
    </script>
    """, unsafe_allow_html=True)

def create_markdown_editor(content, file_path):
    """Create enhanced markdown editor with proper toolbar"""
    
    # Toolbar HTML
    toolbar_html = '''
    <div class="markdown-editor">
        <div class="editor-toolbar">
            <button class="toolbar-btn" onclick="insertText('# ', '')" title="Heading (Ctrl+H)">
                <strong>H</strong>
            </button>
            <button class="toolbar-btn" onclick="wrapText('**', '**')" title="Bold (Ctrl+B)">
                <strong>B</strong>
            </button>
            <button class="toolbar-btn" onclick="wrapText('*', '*')" title="Italic (Ctrl+I)">
                <em>I</em>
            </button>
            <button class="toolbar-btn" onclick="wrapText('`', '`')" title="Code (Ctrl+K)">
                &lt;&gt;
            </button>
            <div class="toolbar-separator"></div>
            <button class="toolbar-btn" onclick="insertText('- ', '')" title="List">
                • List
            </button>
            <button class="toolbar-btn" onclick="insertText('1. ', '')" title="Numbered List">
                1. List
            </button>
            <button class="toolbar-btn" onclick="insertText('> ', '')" title="Quote">
                " Quote
            </button>
            <div class="toolbar-separator"></div>
            <button class="toolbar-btn" onclick="insertText('[', '](url)')" title="Link">
                🔗 Link
            </button>
            <button class="toolbar-btn" onclick="insertText('[[', ']]')" title="Wiki Link">
                [[Wiki]]
            </button>
            <button class="toolbar-btn" onclick="insertText('---\\n', '')" title="Horizontal Rule">
                ─ Rule
            </button>
        </div>
    '''
    
    st.markdown(toolbar_html, unsafe_allow_html=True)
    
    # Text area for editing
    edited_content = st.text_area(
        "Content",
        value=content,
        height=400,
        key=f"editor_{file_path}",
        label_visibility="collapsed"
    )
    
    # Status bar
    lines = len(edited_content.split('\n'))
    chars = len(edited_content)
    words = len(edited_content.split())
    
    st.markdown(f'''
    <div class="status-bar">
        Lines: {lines} | Words: {words} | Characters: {chars} | File: {file_path}
    </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # JavaScript for toolbar functionality
    st.markdown("""
    <script>
    function getEditor() {
        return document.querySelector('textarea[data-testid="stTextArea"] textarea') || 
               document.querySelector('.stTextArea textarea');
    }
    
    function insertText(before, after) {
        const editor = getEditor();
        if (!editor) return;
        
        const start = editor.selectionStart;
        const end = editor.selectionEnd;
        const selectedText = editor.value.substring(start, end);
        
        const newText = before + selectedText + after;
        editor.value = editor.value.substring(0, start) + newText + editor.value.substring(end);
        
        // Set cursor position
        const newPos = start + before.length + selectedText.length;
        editor.setSelectionRange(newPos, newPos);
        editor.focus();
        
        // Trigger change event
        const event = new Event('input', { bubbles: true });
        editor.dispatchEvent(event);
    }
    
    function wrapText(before, after) {
        const editor = getEditor();
        if (!editor) return;
        
        const start = editor.selectionStart;
        const end = editor.selectionEnd;
        const selectedText = editor.value.substring(start, end);
        
        if (selectedText) {
            insertText(before, after);
        } else {
            insertText(before + 'text' + after, '');
            // Select the placeholder text
            setTimeout(() => {
                const newStart = start + before.length;
                const newEnd = newStart + 4; // length of 'text'
                editor.setSelectionRange(newStart, newEnd);
            }, 10);
        }
    }
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey || e.metaKey) {
            switch(e.key) {
                case 'b':
                    e.preventDefault();
                    wrapText('**', '**');
                    break;
                case 'i':
                    e.preventDefault();
                    wrapText('*', '*');
                    break;
                case 'k':
                    e.preventDefault();
                    wrapText('`', '`');
                    break;
                case 'h':
                    e.preventDefault();
                    insertText('# ', '');
                    break;
            }
        }
    });
    </script>
    """, unsafe_allow_html=True)
    
    return edited_content

def render_markdown_preview(content):
    """Render markdown content with wiki link support"""
    # Configure markdown with extensions
    md = markdown.Markdown(
        extensions=['codehilite', 'tables', 'toc', 'fenced_code'],
        extension_configs={
            'codehilite': {
                'css_class': 'highlight',
                'use_pygments': False
            }
        }
    )
    
    # Process wiki links before markdown conversion
    def process_wiki_links(text):
        def wiki_link_replacer(match):
            link_text = match.group(1)
            encoded_link = urllib.parse.quote(link_text)
            return f'<a href="javascript:void(0)" class="wiki-link" onclick="selectWikiLink(\'{encoded_link}\')">{link_text}</a>'
        
        return re.sub(r'\[\[(.*?)\]\]', wiki_link_replacer, text)
    
    # Process content
    processed_content = process_wiki_links(content)
    html_content = md.convert(processed_content)
    
    # Wrap in preview container
    preview_html = f'''
    <div class="markdown-preview">
        {html_content}
    </div>
    
    <script>
    function selectWikiLink(linkText) {
        const decodedLink = decodeURIComponent(linkText);
        console.log('Wiki link clicked:', decodedLink);
        
        // Update URL to trigger Streamlit rerun
        const url = new URL(window.location);
        url.searchParams.set('node', decodedLink);
        window.location.href = url.href;
    }
    </script>
    '''
    
    st.markdown(preview_html, unsafe_allow_html=True)

def extract_links(content):
    """Extract wiki links from content"""
    return re.findall(r'\[\[(.*?)\]\]', content)

def build_enhanced_graph(vault_path, selected_node=None):
    """Build enhanced graph with better visualization"""
    G = nx.Graph()
    files = list(Path(vault_path).rglob("*.md"))
    
    global folder_colors
    
    # Generate colors for new folders
    folders = set(str(f.relative_to(vault_path).parent) for f in files)
    for folder in folders:
        if folder not in folder_colors:
            folder_colors[folder] = f"#{random.randint(0, 0xFFFFFF):06x}"
    
    # Save updated colors to YAML
    save_folder_colors(folder_colors)
    
    # Add nodes with enhanced styling
    for file in files:
        stem = file.stem
        rel_path = str(file.relative_to(vault_path))
        folder = str(file.relative_to(vault_path).parent)
        color = folder_colors.get(folder, "#cccccc")
        
        # Count connections for node size
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        links = extract_links(content)
        node_size = min(50, max(20, len(links) * 5 + 20))
        
        node_attrs = {
            "title": f"{rel_path}\nConnections: {len(links)}",
            "color": color,
            "size": node_size,
            "font": {"size": 14, "color": "#333333"},
            "borderWidth": 2,
            "borderColor": "#666666"
        }
        
        # Highlight selected node
        if stem == selected_node:
            node_attrs.update({
                "borderWidth": 4,
                "borderColor": "#ff4444",
                "size": node_size + 10
            })
        
        G.add_node(stem, **node_attrs)
    
    # Add edges with weights
    edge_weights = {}
    for file in files:
        src_stem = file.stem
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        links = extract_links(content)
        
        for link in links:
            if link in G:
                edge_key = tuple(sorted([src_stem, link]))
                edge_weights[edge_key] = edge_weights.get(edge_key, 0) + 1
    
    # Add edges with visual weight
    for (node1, node2), weight in edge_weights.items():
        G.add_edge(node1, node2, weight=weight, width=min(10, weight * 2))
    
    return G

def create_interactive_graph(graph, height="600px"):
    """Create interactive graph with click handling"""
    try:
        net = Network(height=height, width='100%', bgcolor="#fafafa", font_color="black")
        net.from_nx(graph)
        
        # Configure physics
        net.set_options("""
        var options = {
            "physics": {
                "enabled": true,
                "stabilization": {"iterations": 100}
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 200
            }
        }
        """)
        
        # Save graph
        net.save_graph("graph.html")
        
        # Read and modify HTML
        with open("graph.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        
        # Enhanced click handler
        click_handler = """
        <script>
        document.addEventListener("DOMContentLoaded", function() {
            // Wait for network to be ready
            setTimeout(function() {
                if (typeof network !== 'undefined') {
                    network.on("click", function (params) {
                        if (params.nodes.length > 0) {
                            const nodeId = params.nodes[0];
                            console.log('Graph node clicked:', nodeId);
                            
                            // Update URL to trigger page reload
                            const url = new URL(window.location);
                            url.searchParams.set('node', encodeURIComponent(nodeId));
                            window.location.href = url.href;
                        }
                    });
                    
                    network.on("hoverNode", function (params) {
                        document.body.style.cursor = 'pointer';
                    });
                    
                    network.on("blurNode", function (params) {
                        document.body.style.cursor = 'default';
                    });
                }
            }, 1000);
        });
        </script>
        """
        
        # Inject click handler
        html_content = html_content.replace("</body>", click_handler + "</body>")
        
        # Remove default margins
        html_content = html_content.replace(
            '<body>',
            '<body style="margin:0; padding:10px; font-family: Arial, sans-serif;">'
        )
        
        # Save modified HTML
        with open("graph.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        
        # Display in container
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        with open("graph.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=int(height.replace("px", "")))
        st.markdown('</div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Graph generation failed: {str(e)}")
        st.code(str(e))

# Handle URL parameters for navigation
def handle_navigation():
    """Handle navigation from URL parameters"""
    query_params = st.query_params
    
    # Handle node clicks from graph
    if "node" in query_params:
        node_name = urllib.parse.unquote(query_params["node"])
        # Find file with matching stem
        for file in VAULT_PATH.rglob("*.md"):
            if file.stem == node_name:
                st.session_state.selected_file = str(file.relative_to(VAULT_PATH))
                st.session_state.edit_mode = False
                st.query_params.clear()
                st.rerun()
                break
    
    # Handle direct file selection
    if "file" in query_params:
        file_path = urllib.parse.unquote(query_params["file"])
        if (VAULT_PATH / file_path).exists():
            st.session_state.selected_file = file_path
            st.session_state.edit_mode = False
            st.query_params.clear()
            st.rerun()

# Call navigation handler
handle_navigation()

# Sidebar with enhanced file tree
with st.sidebar:
    st.markdown("### 📁 GraphIQ Explorer")
    
    # File tree buttons for testing (replace with proper event handling)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", key="refresh_tree"):
            st.rerun()
    with col2:
        if st.button("📊 Graph", key="show_graph"):
            st.session_state.graph_update_trigger += 1
    
    # VSCode-style file tree
    render_vscode_file_tree()
    
    st.markdown("---")
    
    # New file creation
    with st.expander("➕ Create New File"):
        with st.form("new_file_form"):
            new_file_name = st.text_input("File Name", placeholder="my-note")
            folder_path = st.text_input("Folder (optional)", placeholder="folder/subfolder")
            submitted = st.form_submit_button("Create File")
            
            if submitted and new_file_name:
                if folder_path:
                    file_dir = VAULT_PATH / folder_path
                    file_dir.mkdir(parents=True, exist_ok=True)
                    new_file_path = file_dir / f"{new_file_name}.md"
                else:
                    new_file_path = VAULT_PATH / f"{new_file_name}.md"
                
                # Create file with template
                template_content = f"# {new_file_name}\n\nCreated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                new_file_path.write_text(template_content, encoding="utf-8")
                
                st.session_state.selected_file = str(new_file_path.relative_to(VAULT_PATH))
                st.success(f"Created: {new_file_name}.md")
                st.rerun()

# Main content area
if st.session_state.selected_file:
    file_path = VAULT_PATH / st.session_state.selected_file
    
    if file_path.exists():
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # File header
            st.markdown(f"### 📄 {file_path.stem}")
            
            # Mode toggle
            mode_col1, mode_col2, mode_col3 = st.columns([1, 1, 2])
            with mode_col1:
                if st.button("✏️ Edit", disabled=st.session_state.edit_mode):
                    st.session_state.edit_mode = True
                    st.rerun()
            
            with mode_col2:
                if st.button("👁️ Preview", disabled=not st.session_state.edit_mode):
                    st.session_state.edit_mode = False
                    st.rerun()
            
            with mode_col3:
                st.caption(f"📁 {file_path.parent.name} / {file_path.name}")
            
            # Content area
            if st.session_state.edit_mode:
                content = file_path.read_text(encoding="utf-8")
                edited_content = create_markdown_editor(content, file_path.name)
                
                # Save controls
                save_col1, save_col2 = st.columns([1, 1])
                with save_col1:
                    if st.button("💾 Save Changes", type="primary"):
                        file_path.write_text(edited_content, encoding="utf-8")
                        st.success("✅ File saved!")
                        st.session_state.edit_mode = False
                        st.session_state.graph_update_trigger += 1  # Trigger graph update
                        time.sleep(0.5)  # Brief pause for user feedback
                        st.rerun()
                
                with save_col2:
                    if st.button("❌ Cancel"):
                        st.session_state.edit_mode = False
                        st.rerun()
            else:
                # Preview mode
                content = file_path.read_text(encoding="utf-8")
                render_markdown_preview(content)
        
        with col2:
            st.markdown("### 🕸️ Knowledge Graph")
            
            # Graph controls
            graph_col1, graph_col2 = st.columns(2)
            with graph_col1:
                show_labels = st.checkbox("Show Labels", value=True)
            with graph_col2:
                graph_height = st.selectbox("Height", ["400px", "500px", "600px", "700px"], index=2)
            
            # Build and display graph
            selected_node = Path(st.session_state.selected_file).stem
            G = build_enhanced_graph(VAULT_PATH, selected_node)
            
            if len(G.nodes()) > 0:
                create_interactive_graph(G, graph_height)
                
                # Graph statistics
                with st.expander("📊 Graph Statistics"):
                    st.metric("Total Notes", len(G.nodes()))
                    st.metric("Connections", len(G.edges()))
                    
                    if selected_node in G:
                        neighbors = list(G.neighbors(selected_node))
                        st.metric("Connected Notes", len(neighbors))
                        if neighbors:
                            st.write("**Connected to:**")
                            for neighbor in neighbors[:5]:  # Show first 5
                                st.write(f"• {neighbor}")
                            if len(neighbors) > 5:
                                st.write(f"... and {len(neighbors) - 5} more")
            else:
                st.info("📝 Create more notes with [[wiki links]] to see connections in the graph!")
    else:
        st.error(f"File not found: {st.session_state.selected_file}")
        st.session_state.selected_file = None
        st.rerun()
else:
    # Welcome screen
    st.markdown("""
    # 🧠 GraphIQ
    ### Your Personal Knowledge Graph
    
    Welcome to GraphIQ! This is an Obsidian-inspired markdown editor with powerful graph visualization.
    
    **🚀 Getting Started:**
    1. Create your first note using the sidebar
    2. Write in Markdown and use `[[wiki links]]` to connect ideas
    3. Watch your knowledge graph grow automatically
    4. Click on graph nodes to navigate between notes
    
    **✨ Features:**
    - 📝 Rich markdown editor with toolbar
    - 🔗 Wiki-style linking between notes
    - 📊 Interactive knowledge graph
    - 📁 VSCode-like file explorer
    - 🎨 Folder color coding
    - ⌨️ Keyboard shortcuts (Ctrl+B, Ctrl+I, etc.)
    
    **Select a file from the sidebar to begin your knowledge journey!**
    """)
    
    # Sample files suggestion
    if not any(VAULT_PATH.rglob("*.md")):
        st.markdown("---")
        if st.button("🎯 Create Sample Notes"):
