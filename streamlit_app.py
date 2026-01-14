# streamlit_app.py - Fixed for your exact file structure
import sys
import os
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent.absolute()
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

print(f"📍 Project root: {project_root}")
print(f"📁 Added to Python path: {src_path}")
print(f"🔧 Python will look for packages in: {src_path}")

# Now import normally - these should work!
try:
    from agents.coordinator_agent import CoordinatorAgent
    from utils.config import load_config
    from utils.logger import setup_logger
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Continue anyway - we'll handle it gracefully
    pass

# Your existing Streamlit imports and code
import streamlit as st
import asyncio
from datetime import datetime
import pandas as pd
from PIL import Image
import io
import json

# Page config
st.set_page_config(
    page_title="Smart Document Processor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        color: #1a1a1a; /* <--- ADD THIS LINE TO FIX THE TEXT COLOR */
    }
    .success-card {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .error-card {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_coordinator():
    """Get or create coordinator instance"""
    try:
        config = load_config()
        coordinator = CoordinatorAgent("DocumentCoordinator", config["agents"]["coordinator"])
        
        # Create event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Setup coordinator
        loop.run_until_complete(coordinator.setup(config))
        
        return coordinator, loop
    except Exception as e:
        st.error(f"Failed to initialize coordinator: {str(e)}")
        return None, None

def init_session_state():
    """Initialize session state variables"""
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'coordinator' not in st.session_state:
        st.session_state.coordinator, st.session_state.event_loop = get_coordinator()

def main():
    """Main Streamlit app"""
    init_session_state()
    
    # Header
    st.title("🤖 Smart Document Processor")
    st.markdown("**Multi-Agent Document Processing with ERNIE & PaddleOCR**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        extraction_type = st.selectbox(
            "Extraction Type",
            ["text", "table", "structure", "full"],
            help="Choose what to extract from the document"
        )
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["auto", "invoice", "contract", "form", "report", "general"],
            help="Type of document analysis"
        )
        
        workflow_type = st.selectbox(
            "Workflow Type",
            ["standard", "parallel", "streaming"],
            help="Processing workflow to use"
        )
        
        if st.button("🔄 Refresh System Status"):
            if st.session_state.coordinator:
                st.success("System status refreshed!")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📄 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['pdf', 'jpg', 'jpeg', 'png', 'tiff', 'bmp', 'docx'],
            help="Upload a document to process"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"📊 Size: {uploaded_file.size / 1024:.1f} KB")
        
        process_button = st.button("🚀 Process Document", disabled=uploaded_file is None)
    
    with col2:
        st.header("📊 System Status")
        
        if st.session_state.coordinator:
            status = st.session_state.coordinator.get_all_agents_status()
            
            # Display agent statuses
            for agent_name in ["ocr", "analysis", "validation"]:
                agent_status = status[agent_name]
                status_color = "🟢" if agent_status["status"] == "completed" else "🔴"
                
                with st.container():
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{status_color} {agent_name.upper()}</strong><br>
                        Status: {agent_status["status"]}<br>
                        Tasks: {agent_status["metrics"]["total_tasks_processed"]}<br>
                        Success: {agent_status["metrics"]["success_rate"]:.1%}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("❌ System not initialized")
    
    # Processing
    if process_button and uploaded_file and st.session_state.coordinator:
        with st.spinner("🔄 Processing document... This may take a moment."):
            try:
                # Save uploaded file temporarily
                temp_dir = Path("temp_uploads")
                temp_dir.mkdir(exist_ok=True)
                
                file_path = temp_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Create processing task
                task = {
                    "type": "document_processing",
                    "document_path": str(file_path),
                    "extraction_type": extraction_type,
                    "output_format": "json",
                    "analysis_type": analysis_type,
                    "workflow_type": workflow_type,
                    "original_filename": uploaded_file.name
                }
                
                # Process document
                loop = st.session_state.event_loop
                result = loop.run_until_complete(st.session_state.coordinator.process(task))
                
                # Clean up temp file
                try:
                    file_path.unlink()
                except:
                    pass
                
                if result["success"]:
                    st.session_state.current_result = result["result"]
                    st.success("✅ Document processed successfully!")
                else:
                    st.error(f"❌ Processing failed: {result.get('error', {}).get('message', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
    
    # Results display
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        st.header("📋 Processing Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📝 Extracted Text", "🔍 Analysis", "✅ Validation"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                doc_info = result.get("document_information", {})
                st.markdown("**📄 Document Info**")
                st.write(f"Filename: {doc_info.get('filename', 'Unknown')}")
                st.write(f"Size: {doc_info.get('file_size', 0)} MB")
                st.write(f"Pages: {doc_info.get('pages', 0)}")
            
            with col2:
                analysis = result.get("analysis_results", {})
                st.markdown("**🔍 Analysis**")
                st.write(f"Type: {analysis.get('document_type', 'Unknown')}")
                st.write(f"Category: {analysis.get('category', 'General')}")
                st.write(f"Confidence: {analysis.get('analysis_confidence', 0):.2f}")
            
            with col3:
                validation = result.get("validation_results", {})
                quality = result.get("quality_metrics", {})
                st.markdown("**✅ Quality**")
                st.write(f"Valid: {'✅' if validation.get('is_valid') else '❌'}")
                st.write(f"Quality Score: {quality.get('overall_quality_score', 0):.2f}")
                st.write(f"Level: {quality.get('quality_level', 'Unknown')}")
            
            # Summary
            st.markdown("**📝 Summary**")
            st.info(analysis.get('summary', 'No summary available'))
        
        with tab2:
            extracted_text = result.get("extraction_results", {}).get("extracted_content", "")
            st.text_area("Extracted Text", extracted_text, height=300)
            
            # Text statistics
            stats = result.get("text_statistics", {})
            if stats:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Characters", stats.get("total_characters", 0))
                with col2:
                    st.metric("Words", stats.get("total_words", 0))
                with col3:
                    st.metric("Lines", stats.get("total_lines", 0))
        
        with tab3:
            # Entities
            entities = result.get("analysis_results", {}).get("entities", {})
            if entities:
                st.markdown("**🏷️ Extracted Entities**")
                
                for entity_type, entity_list in entities.items():
                    if entity_list:
                        with st.expander(f"{entity_type.title()} ({len(entity_list)})"):
                            for entity in entity_list:
                                st.write(f"• {entity}")
            
            # Detailed analysis
            detailed = result.get("analysis_results", {}).get("detailed_analysis", {})
            if detailed:
                st.markdown("**🔍 Detailed Analysis**")
                st.json(detailed)
        
        with tab4:
            validation = result.get("validation_results", {})
            
            # Validation score
            score = validation.get("validation_score", 0)
            st.progress(score)
            st.write(f"Validation Score: {score:.2f}")
            
            # Issues
            issues = validation.get("issues", {})
            if issues:
                st.markdown("**⚠️ Issues Found**")
                for issue_type, issue_list in issues.items():
                    if issue_list:
                        with st.expander(f"{issue_type.title()} ({len(issue_list)})"):
                            for issue in issue_list:
                                st.write(f"• {issue}")
            
            # Recommendations
            recommendations = validation.get("recommendations", [])
            if recommendations:
                st.markdown("**💡 Recommendations**")
                for rec in recommendations:
                    st.write(f"• {rec}")
        
        # Export options
        st.header("💾 Export Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📥 Export JSON"):
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(result, indent=2, ensure_ascii=False),
                    file_name=f"results_{Path(result['document_information']['filename']).stem}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📄 Export Markdown"):
                markdown_content = f"# Document Processing Results\n\n## Summary\n{analysis.get('summary', '')}\n\n## Document Info\n- **Filename**: {doc_info.get('filename', '')}\n- **Type**: {analysis.get('document_type', '')}\n- **Confidence**: {result['metadata']['overall_confidence']:.2f}\n\n## Extracted Text\n```\n{result.get('extraction_results', {}).get('extracted_content', '')[:500]}...\n```"
                st.download_button(
                    label="Download Markdown",
                    data=markdown_content,
                    file_name=f"results_{Path(result['document_information']['filename']).stem}.md",
                    mime="text/markdown"
                )
        
        with col3:
            if st.button("📊 Export CSV"):
                # Create CSV from entities
                csv_data = []
                entities = result.get("analysis_results", {}).get("entities", {})
                for entity_type, entity_list in entities.items():
                    for entity in entity_list:
                        csv_data.append({"Type": entity_type, "Entity": entity})
                
                if csv_data:
                    df = pd.DataFrame(csv_data)
                    st.download_button(
                        label="Download CSV",
                        data=df.to_csv(index=False),
                        file_name=f"results_{Path(result['document_information']['filename']).stem}.csv",
                        mime="text/csv"
                    )
        
        with col4:
            if st.button("🌐 Export HTML"):
                html_content = f"<html><body><h1>Document Processing Results</h1><h2>Summary</h2><p>{analysis.get('summary', '')}</p><h2>Document Information</h2><ul><li>Filename: {doc_info.get('filename', '')}</li><li>Type: {analysis.get('document_type', '')}</li><li>Confidence: {result['metadata']['overall_confidence']:.2f}</li></ul><h2>Extracted Text</h2><pre>{result.get('extraction_results', {}).get('extracted_content', '')[:1000]}...</pre></body></html>"
                st.download_button(
                    label="Download HTML",
                    data=html_content,
                    file_name=f"results_{Path(result['document_information']['filename']).stem}.html",
                    mime="text/html"
                )
    
    # Processing history
    if st.session_state.processing_history:
        st.header("📜 Processing History")
        
        for i, history_item in enumerate(reversed(st.session_state.processing_history[-5:])):
            with st.expander(f"📄 {history_item['filename']} - {history_item['timestamp']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"Type: {history_item['document_type']}")
                with col2:
                    st.write(f"Confidence: {history_item['confidence']:.2f}")
                with col3:
                    st.write(f"Processing Time: {history_item['processing_time']:.1f}s")

if __name__ == "__main__":
    main()