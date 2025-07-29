Enhanced Markdown Editing

Instead of the built-in st.text_area, use a richer editor component. For example, Streamlit-Monaco wraps VS Code’s Monaco editor and supports a Markdown mode. It is MIT-licensed and easy to install (pip install streamlit-monaco). In code you can do:

from streamlit_monaco import st_monaco
content = st_monaco(value="# Hello world", language="markdown")  # Returns the edited markdown

which gives you a full-featured editor with syntax highlighting.  Similarly, Streamlit-Ace provides the Ace code editor (pip install streamlit-ace) and can be used for Markdown. Its basic usage is:

from streamlit_ace import st_ace
content = st_ace(language="markdown")

(see [29] for example).  For a true WYSIWYG editor, Streamlit-Quill (based on the Quill rich-text editor) can be embedded with st_quill(), returning HTML output. These components all allow more interactive editing (toolbars, shortcuts, etc.) than the simple text area.

Fixing Link/Node Click Behavior

To make wiki-links and graph clicks reliably update the view, use URL query parameters and Streamlit’s state.  For example, wrap [[Link]] occurrences in <a href="?node=PageName"> so that clicking sets ?node= in the URL. In Python use st.query_params to read this parameter and set st.session_state.selected_file accordingly. For instance:

params = st.query_params
if "node" in params:
    st.session_state.selected_file = urllib.parse.unquote(params["node"][0])
    st.query_params.clear()
    st.experimental_rerun()

Streamlit’s st.query_params lets you read/write URL parameters as shown in the docs.  Likewise, for the PyVis graph you should change the injected JavaScript to update the query string instead of the hash. For example:

// Inside the PyVis-generated HTML
node.addEventListener("click", function() {
  const title = this.getAttribute("title");
  // Redirect to same page with ?node=<title>
  window.location.search = '?node=' + encodeURIComponent(title);
});

By setting window.location.search, Streamlit will reload with that query param, and your code can respond (via st.query_params) to highlight or open that file. In short, use st.query_params (or st.experimental_set_query_params) on click so that every node/link click causes the app to rerun with the appropriate file loaded.

Improved File Explorer UI

Replace the manual st.button list with a proper tree component. One option is streamlit-file-browser (MIT-licensed) which provides a collapsible folder view and file selection out-of-the-box.  For example:

from streamlit_file_browser import st_file_browser
event = st_file_browser("vault", key="file_browser")
if event and event["is_file"]:
    st.session_state.selected_file = event["file_path"]

This shows a VSCode-like file tree (folders and markdown files), with an option to preview or download.  Installing it is simple (pip install streamlit-file-browser), and the usage example in the README is:

event = st_file_browser("example_artifacts", key='A')
st.write(event)

which yields the selected path.  Other options include using a checkbox-tree component (e.g. streamlit-tree-select [49]) or using st.expander nesting manually, but st_file_browser already implements collapsible folders, file icons, and click events in one component.  It will feel much more dynamic than buttons.

Vault YAML Schema

Define vault.yml as a hierarchy of folders, each with files. For example:

folders:
  - name: Root
    files:
      - name: Introduction.md
        content: | 
          100–200 words of markdown text covering the introduction...
        links: [ConceptA, ConceptB]
      - name: Overview.md
        content: | 
          100–200 words of markdown text for an overview...
        links: [ConceptA]
    subfolders:
      - name: Details
        files:
          - name: Details.md
            content: |
              100–200 words on a specific detail...
            links: [Overview, ConceptC]
        subfolders: []
  - name: Appendix
    files:
      - name: References.md
        content: |
          100–200 words with references or further reading...
        links: []
    subfolders: []

In this schema: each folder has a name, an optional list of subfolders, and a list of files. Each file entry has a name (including “.md”), a content block of markdown text (~100–200 words), and a list of wiki-style links to other note titles.  The hierarchy can be nested arbitrarily via the subfolders field.  (This structure can be extended as needed, e.g. adding folder-level color codes or metadata.)

LLM Prompt Template for Vault Generation

You can prompt an LLM to generate a complete vault.yml for any topic by giving it the schema and requirements. For example:

> Prompt: Generate a vault.yml structure for learning about “Blockchain”, using the schema above. The YAML should define folders, files, file contents, and links between notes. Each file content must be 100–200 words of useful markdown on the subtopic. Include pages like “Introduction.md”, “Consensus.md”, etc., and use wiki-links (in the links lists) to connect related pages. Output valid YAML following the folder/files schema.



The response should be pure YAML. You can adapt the prompt for any topic by replacing “Blockchain” with your desired subject. This instructs the LLM to produce a hierarchical vault with meaningful contents and inter-page links, which your app can then visualize as a graph.

Sources: Streamlit editor components and file-browser documentation. These illustrate usage of Monaco/Ace/Quill editors and query-parameter navigation in Streamlit.

