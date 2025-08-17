import streamlit as st
from dotenv import load_dotenv
import os
import time
import asyncio
import threading
from supabase import create_client, Client
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from semantic_chunking import ChunkingManager
# --- Thay đổi 1: Import các lớp từ Google ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai # Thêm import này
from langchain.prompts import PromptTemplate
import cohere
from langchain_cohere import CohereRerank
from langchain.chains.combine_documents import create_stuff_documents_chain

# Import authentication modules
from auth import AuthManager, AuthUI



# ===== STREAMLIT PAGE CONFIG =====
st.set_page_config(
    page_title="🤖 AI Trợ Giảng Toán Tin",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/chatbot',
        'Report a bug': "https://github.com/yourusername/chatbot/issues",
        'About': "# AI Trợ Giảng Toán Tin\nPowered by Gemini & Supabase"
    }
)

# ===== CUSTOM CSS STYLING =====
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styling */
.stApp {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Main Container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Header Styling */
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.main-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    animation: gradient 3s ease-in-out infinite;
}

.main-subtitle {
    font-size: 1.2rem;
    color: rgba(255, 255, 255, 0.8);
    font-weight: 400;
}

@keyframes gradient {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* Sidebar Styling */
.css-1d391kg {
    background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%);
}

.sidebar-header {
    background: linear-gradient(135deg, #FF6B6B, #4ECDC4);
    padding: 1.5rem;
    border-radius: 15px;
    margin-bottom: 1.5rem;
    text-align: center;
    color: white;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

/* File Uploader Styling */
.stFileUploader {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 1rem;
    border: 2px dashed rgba(255, 255, 255, 0.3);
    transition: all 0.3s ease;
}

.stFileUploader:hover {
    border-color: #4ECDC4;
    background: rgba(78, 205, 196, 0.1);
    transform: translateY(-2px);
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

/* Chat Messages */
.stChatMessage {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 1rem;
    margin: 0.5rem 0;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    animation: fadeIn 0.5s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* User Message */
.stChatMessage[data-testid="user-message"] {
    background: linear-gradient(135deg, #FF6B6B, #FF8E53);
    margin-left: 20%;
}

/* Assistant Message */
.stChatMessage[data-testid="assistant-message"] {
    background: linear-gradient(135deg, #4ECDC4, #44A08D);
    margin-right: 20%;
}

/* Chat Input */
.stChatInput > div > div > input {
    background: rgba(255, 255, 255, 0.1);
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-radius: 25px;
    color: white;
    font-size: 1rem;
    padding: 1rem 1.5rem;
    backdrop-filter: blur(10px);
}

.stChatInput > div > div > input:focus {
    border-color: #4ECDC4;
    box-shadow: 0 0 20px rgba(78, 205, 196, 0.3);
}

/* Success/Error Messages */
.stSuccess {
    background: linear-gradient(135deg, #56ab2f, #a8e6cf);
    border-radius: 15px;
    border: none;
    color: white;
    font-weight: 500;
}

.stError {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    border-radius: 15px;
    border: none;
    color: white;
    font-weight: 500;
}

/* Spinner */
.stSpinner {
    text-align: center;
}

/* Stats Cards */
.stats-card {
    background: rgba(255, 255, 255, 0.1);
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    margin: 0.5rem;
    transition: transform 0.3s ease;
}

.stats-card:hover {
    transform: translateY(-5px);
}

.stats-number {
    font-size: 2rem;
    font-weight: 700;
    color: #4ECDC4;
}

.stats-label {
    font-size: 0.9rem;
    color: rgba(255, 255, 255, 0.7);
    margin-top: 0.5rem;
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-title {
        font-size: 2rem;
    }
    
    .stChatMessage[data-testid="user-message"] {
        margin-left: 10%;
    }
    
    .stChatMessage[data-testid="assistant-message"] {
        margin-right: 10%;
    }
}
</style>
""", unsafe_allow_html=True)

# Tải các biến môi trường từ file .env
load_dotenv()

def stream_parser(stream):
    """
    Lắng nghe một stream từ LangChain và chỉ yield (đẩy ra) phần nội dung.
    """
    for chunk in stream:
        # Với document_chain, chunk là string trực tiếp
        if isinstance(chunk, str):
            yield chunk
        # Với ConversationalRetrievalChain, chunk có key "answer"
        elif isinstance(chunk, dict) and "answer" in chunk:
            yield chunk["answer"]

def run_async_in_thread(async_func, *args):
    """
    Chạy hàm async trong thread riêng để tránh lỗi event loop.
    """
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_func(*args))
        finally:
            loop.close()
    
    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join()
    return thread.result if hasattr(thread, 'result') else None

def safe_stream_qa(qa_chain, prompt):
    """
    Wrapper an toàn cho streaming QA để tránh lỗi event loop.
    """
    try:
        # Thử streaming trước
        return qa_chain.stream(prompt)
    except Exception as e:
        if "event loop" in str(e).lower():
            # Nếu gặp lỗi event loop, dùng invoke thay thế
            st.warning("Đang sử dụng chế độ đồng bộ do giới hạn kỹ thuật...")
            result = qa_chain.invoke(prompt)
            # Tạo generator giả để tương thích với stream_parser
            def fake_stream():
                yield {"answer": result.get("answer", "")}
            return fake_stream()
        else:
            raise e

# --- Khởi tạo với xử lý event loop ---
def initialize_components():
    """Khởi tạo các thành phần với xử lý event loop an toàn."""
    try:
        # Đảm bảo có event loop cho thread hiện tại
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        google_api_key = os.getenv("GOOGLE_API_KEY")
        cohere_api_key = os.getenv("COHERE_API_KEY")

        if not cohere_api_key:
            st.error("Vui lòng thiết lập biến môi trường COHERE_API_KEY trong file .env")
            st.stop()
        #cohere.configure(api_key=cohere_api_key)
        if not google_api_key:
            st.error("Vui lòng thiết lập biến môi trường GOOGLE_API_KEY trong file .env")
            st.stop()
        genai.configure(api_key=google_api_key)

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if not all([supabase_url, supabase_key]):
            st.error("Vui lòng thiết lập các biến môi trường SUPABASE_URL, và SUPABASE_KEY trong file .env")
            st.stop()

        supabase: Client = create_client(supabase_url, supabase_key)

        # Khởi tạo embeddings với xử lý lỗi
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        vector_store = SupabaseVectorStore(
            client=supabase, 
            table_name="documents", 
            embedding=embeddings, 
            query_name="match_documents"
        )
        
        # Khởi tạo LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=1.0, 
           # convert_system_message_to_human=True
        )
        
        return supabase, embeddings, vector_store, llm
        
    except Exception as e:
        st.error(f"Lỗi khởi tạo: {e}")
        st.stop()

# Khởi tạo components
try:
    supabase, embeddings, vector_store, llm = initialize_components()
    
    # Khởi tạo Authentication
    auth_manager = AuthManager(supabase)
    auth_ui = AuthUI(auth_manager)
    
    # Khởi tạo Re-Ranker
    reranker = CohereRerank(
        model="rerank-multilingual-v3.0",
        top_n=5
    )
    
    # Khởi tạo Semantic Chunking Manager
    chunking_manager = ChunkingManager(embeddings)
    # Thiết lập prompt
    prompt_template = ("""
                       Bạn là một trợ giảng AI chuyên ngành cho sinh viên đại học, thân thiện và cực kỳ cẩn thận. Nhiệm vụ của bạn là giúp sinh viên hiểu sâu các khái niệm, giải bài tập và ôn tập dựa trên tài liệu học tập của họ.

                       NHIỆM VỤ: Dựa vào dữ liệu trong tài liệu, trả lời trực tiếp, nếu không biết thì nói 'không có trong dữ liệu'

                       HƯỚNG DẪN:
                       1. Đọc kỹ câu hỏi
                       2. Nếu thông tin có trong tài liệu: Trả lời chi tiết, rõ ràng
                       3. Nếu không có thông tin: Nói rằng "Tôi không tìm thấy thông tin này trong tài liệu đã cung cấp"
                       4. Sử dụng Markdown để format đẹp
                       
                       Ngữ cảnh tài liệu:
                       {context}

                       Câu hỏi: {question}
                       
                       Lịch sử trò chuyện:
                       {chat_history}

                       Trả lời:""")
    
    QA_prompt = PromptTemplate.from_template(prompt_template)
                        
    # Thiết lập bộ nhớ để lưu trữ lịch sử cuộc trò chuyện
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    # Chain nhỏ để kết hợp tài liệu vào prompt
    document_chain = create_stuff_documents_chain(llm, QA_prompt)

    # Chain lớn hơn
    #conversation_retrieval_chain = create_retrieval_chain(retriever=vector_store.as_retriever(search_kwargs={"k": 7}), document_chain=document_chain)

    # Tạo chuỗi xử lý hội thoại
    #qa = ConversationalRetrievalChain.from_llm(llm, retriever=vector_store.as_retriever(search_kwargs={"k": 7}), memory=memory, combine_docs_chain_kwargs={"prompt": QA_prompt})
    #qa = conversation_retrieval_chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 5, "score_threshold": 0.5})
except Exception as e:
    st.error(f"Lỗi khởi tạo: {e}")
    st.stop()


def process_document(uploaded_file, chunking_strategy="adaptive"):
    """
    Hàm xử lý file được tải lên: đọc, cắt nhỏ thông minh và lưu embeddings vào Supabase.
    """
    try:
        # Tạo thư mục temp nếu chưa có
        temp_dir = "./temp_files"
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(temp_file_path)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            loader = Docx2txtLoader(temp_file_path)
        else:
            loader = TextLoader(temp_file_path)
        
        documents = loader.load()
        for doc in documents:
            doc.page_content = doc.page_content.replace('\u0000', '')

        # Sử dụng Semantic Chunking thay vì basic chunking
        st.write(f"🧠 **Đang áp dụng {chunking_strategy} chunking...**")
        docs, chunk_stats = chunking_manager.process_documents(documents, chunking_strategy)
        
        # Hiển thị thống kê chunking chi tiết
        st.write(f"📊 **Thống kê chunking:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tổng chunks", chunk_stats.get('total_chunks', 0))
        with col2:
            st.metric("Độ dài TB", f"{chunk_stats.get('avg_chunk_length', 0):.0f}")
        with col3:
            st.metric("Thời gian xử lý", f"{chunk_stats.get('processing_time', 0):.2f}s")
        
        # Hiển thị phân bố loại chunks
        chunk_types = chunk_stats.get('chunk_types', {})
        if chunk_types:
            st.write("📋 **Phân bố loại chunks:**")
            for chunk_type, count in chunk_types.items():
                st.write(f"• {chunk_type}: {count} chunks")
        
        # Add metadata to documents
        for doc in docs:
            doc.metadata["source_file"] = uploaded_file.name
            doc.metadata["upload_time"] = time.time()
            doc.metadata["chunking_strategy"] = chunking_strategy
        
        vector_store.add_documents(docs)
        st.write(f"✅ **Đã lưu {len(docs)} chunks vào vector database**")

        os.remove(temp_file_path)
        return True, chunk_stats
    except Exception as e:
        st.error(f"Lỗi khi xử lý tài liệu: {e}")
        return False, {}

# ===== HELPER FUNCTIONS =====
def display_typing_animation():
    """Hiển thị animation typing với emoji"""
    typing_placeholder = st.empty()
    for i in range(3):
        typing_placeholder.markdown(f"🤖 **Trợ lý AI đang suy nghĩ{'.' * (i + 1)}**")
        time.sleep(0.5)
    typing_placeholder.empty()

def get_file_stats():
    """Lấy thống kê file đã upload"""
    if "uploaded_files_count" not in st.session_state:
        st.session_state.uploaded_files_count = 0
    if "total_messages" not in st.session_state:
        st.session_state.total_messages = len(st.session_state.get("messages", []))
    return st.session_state.uploaded_files_count, st.session_state.total_messages

# ===== AUTHENTICATION CHECK =====
if not auth_ui.render_auth_page():
    st.stop()

# ===== MAIN INTERFACE =====

# Beautiful Header with Animation
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🧠 AI Trợ Giảng Toán Tin</h1>
    <p class="main-subtitle">✨ Powered by Gemini 2.5 Pro & Supabase Vector Store ✨</p>
    <div style="margin-top: 1rem;">
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem; font-size: 0.9rem;">📚 Hỏi đáp thông minh</span>
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem; font-size: 0.9rem;">🚀 Streaming Response</span>
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem; font-size: 0.9rem;">🎯 Context-Aware</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Stats Dashboard
file_count, message_count = get_file_stats()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-number">{file_count}</div>
        <div class="stats-label">📄 Tài liệu</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-number">{message_count}</div>
        <div class="stats-label">💬 Tin nhắn</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stats-card">
        <div class="stats-number">🔥</div>
        <div class="stats-label">⚡ Gemini 2.5</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stats-card">
        <div class="stats-number">✨</div>
        <div class="stats-label">🎨 Modern UI</div>
    </div>
    """, unsafe_allow_html=True)

# ===== ENHANCED SIDEBAR =====
with st.sidebar:
    # User Profile Section
    auth_ui.render_user_profile()
    
    st.markdown("---")
    
    # Sidebar Header
    st.markdown("""
    <div class="sidebar-header">
        <h2 style="margin: 0; font-size: 1.5rem;">📚 Quản lý Tài liệu</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Upload & xử lý tài liệu học tập</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File Upload Section - Admin Only
    if auth_manager.can_upload_documents():
        st.markdown("### 📁 Tải lên tài liệu")
        uploaded_files = st.file_uploader(
            "Kéo thả hoặc chọn file",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Hỗ trợ: PDF, DOCX, TXT. Có thể chọn nhiều file cùng lúc."
        )
        
        # Chunking Strategy Selection
        st.markdown("### 🧠 Chiến lược cắt nhỏ dữ liệu")
        strategies = chunking_manager.get_available_strategies()
        strategy_names = list(strategies.keys())
        strategy_descriptions = [strategies[key] for key in strategy_names]
        
        selected_strategy = st.selectbox(
            "Chọn phương pháp chunking:",
            options=strategy_names,
            index=1,  # Default to 'adaptive'
            format_func=lambda x: f"{x.title()} - {strategies[x]}",
            help="Adaptive sẽ tự động chọn phương pháp tốt nhất cho từng loại nội dung"
        )
        
        # Show strategy description
        st.info(f"📝 **{selected_strategy.title()}**: {strategies[selected_strategy]}")
        
    else:
        st.markdown("### 🔒 Tải lên tài liệu")
        st.info("⚠️ Chỉ có quản trị viên mới có thể tải lên tài liệu.")
        uploaded_files = None
        selected_strategy = "adaptive"
    
    if uploaded_files:
        st.markdown(f"**📊 Đã chọn {len(uploaded_files)} file:**")
        for file in uploaded_files:
            file_size = len(file.getvalue()) / 1024  # KB
            st.markdown(f"• `{file.name}` ({file_size:.1f} KB)")
        
        st.markdown("---")
        
        if st.button("🚀 Xử lý và Nạp kiến thức", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_stats = {'total_chunks': 0, 'total_time': 0, 'strategies_used': []}
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Đang xử lý: {file.name} với {selected_strategy} chunking")
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                success, chunk_stats = process_document(file, selected_strategy)
                if success:
                    st.session_state.uploaded_files_count += 1
                    total_stats['total_chunks'] += chunk_stats.get('total_chunks', 0)
                    total_stats['total_time'] += chunk_stats.get('processing_time', 0)
                    total_stats['strategies_used'].append(chunk_stats.get('strategy_used', selected_strategy))
                    
            progress_bar.empty()
            status_text.empty()
            
            # Show summary statistics
            st.success(f"✅ Đã xử lý thành công {len(uploaded_files)} tài liệu!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Tổng chunks", total_stats['total_chunks'])
            with col2:
                st.metric("⏱️ Tổng thời gian", f"{total_stats['total_time']:.2f}s")
            with col3:
                st.metric("🧠 Chiến lược", selected_strategy.title())
            
            st.balloons()
    
    # Divider
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### ⚡ Thao tác nhanh")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Xóa chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Tips Section
    st.markdown("---")
    st.markdown("### 💡 Mẹo sử dụng")
    st.markdown("""
    • 📝 **Hỏi cụ thể**: "Giải thích thuật toán Dijkstra"
    • 🔍 **Tìm kiếm**: "Tìm định nghĩa về đồ thị"
    • 📊 **So sánh**: "So sánh BFS và DFS"
    • 🧮 **Bài tập**: "Cho ví dụ về cây nhị phân"
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; opacity: 0.7; font-size: 0.8rem;">
        <p>🤖 Made with ❤️ by Duy</p>
        <p>Powered by Streamlit & Gemini</p>
    </div>
    """, unsafe_allow_html=True)

# ===== ENHANCED CHAT INTERFACE =====

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome message for first time users
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 15px; margin: 2rem 0; backdrop-filter: blur(10px);">
        <h3 style="color: #4ECDC4; margin-bottom: 1rem;">👋 Chào mừng đến với AI Trợ Giảng!</h3>
        <p style="color: rgba(255,255,255,0.8); margin-bottom: 1rem;">Tôi là trợ lý AI chuyên ngành Toán Tin, sẵn sàng giúp bạn:</p>
        <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 1rem; margin-top: 1rem;">
            <span style="background: rgba(255,107,107,0.2); color: #FF6B6B; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">🧮 Giải bài tập</span>
            <span style="background: rgba(78,205,196,0.2); color: #4ECDC4; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">📚 Giải thích lý thuyết</span>
            <span style="background: rgba(69,183,209,0.2); color: #45B7D1; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">🔍 Tìm kiếm thông tin</span>
        </div>
        <p style="color: rgba(255,255,255,0.6); font-size: 0.9rem; margin-top: 1rem;">💡 Hãy upload tài liệu và bắt đầu hỏi đáp!</p>
    </div>
    """, unsafe_allow_html=True)

# Display chat messages with enhanced styling
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        if message["role"] == "user":
            st.markdown(f"**Bạn:** {message['content']}")
        else:
            st.markdown(message["content"])

# Enhanced chat input with suggestions
if prompt := st.chat_input("💬 Hãy đặt câu hỏi về tài liệu của bạn...", key="chat_input"):
    chat_history = memory.load_memory_variables({}).get("chat_history", [])

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.total_messages = len(st.session_state.messages)
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"**Bạn:** {prompt}")

    # Display assistant response with enhanced streaming
    with st.chat_message("assistant", avatar="🤖"):
        # Show typing indicator
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("🤖 **Đang phân tích tài liệu và tạo câu trả lời...** ⏳")
        
        try:
            # Debug: Show retrieval process
            st.write("🔍 **Debug - Đang tìm kiếm tài liệu...**")
            retrieved_docs = retriever.invoke(prompt)
            st.write(f"📄 **Tìm thấy {len(retrieved_docs)} tài liệu liên quan**")
            
            # Show chunking strategy info for retrieved docs
            chunk_strategies = {}
            for doc in retrieved_docs:
                strategy = doc.metadata.get('chunking_strategy', 'unknown')
                chunk_strategies[strategy] = chunk_strategies.get(strategy, 0) + 1
            
            if chunk_strategies:
                strategy_info = ", ".join([f"{k}: {v}" for k, v in chunk_strategies.items()])
                st.write(f"🧠 **Chunking strategies: {strategy_info}**")
            
            # Show retrieved documents for debugging
            # with st.expander("📋 Xem tài liệu được tìm thấy"):
            #     for i, doc in enumerate(retrieved_docs):
            #         st.write(f"**Tài liệu {i+1}:**")
            #         st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
            #         st.write("---")

            st.write("🎯 **Đang rerank tài liệu...**")
            reranked_docs = reranker.compress_documents(
                documents=retrieved_docs,
                query=prompt
            )
            st.write(f"✨ **Sau rerank: {len(reranked_docs)} tài liệu tốt nhất**")
            
            # Show final context being sent to LLM
            # with st.expander("📝 Xem context cuối cùng gửi cho AI"):
            #     context_text = "\n\n".join([doc.page_content for doc in reranked_docs])
            #     st.write(f"**Tổng độ dài context:** {len(context_text)} ký tự")
            #     st.text_area("Context:", context_text, height=200)
            
            # Show reranked documents for debugging
            # with st.expander("🏆 Xem tài liệu sau rerank"):
            #     for i, doc in enumerate(reranked_docs):
            #         st.write(f"**Top {i+1}:**")
            #         st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
            #         st.write("---")

            st.write("🤖 **Đang tạo câu trả lời...**")
            
            # Stream the response using safe wrapper
            response_stream = document_chain.stream({
                "question": prompt,
                "context": reranked_docs,
                "chat_history": chat_history
            })
            thinking_placeholder.empty()
            
            # Display streaming response with better formatting
            full_response = st.write_stream(stream_parser(response_stream))
            
            # Add reaction buttons
            col1, col2, col3, col4 = st.columns([1, 1, 1, 6])
            with col1:
                if st.button("👍", key=f"like_{len(st.session_state.messages)}"):
                    st.toast("Cảm ơn phản hồi của bạn! 😊", icon="👍")
            with col2:
                if st.button("👎", key=f"dislike_{len(st.session_state.messages)}"):
                    st.toast("Tôi sẽ cố gắng cải thiện! 🙏", icon="👎")
            with col3:
                if st.button("🔄", key=f"retry_{len(st.session_state.messages)}"):
                    st.rerun()
                    
        except Exception as e:
            thinking_placeholder.empty()
            st.error(f"❌ Có lỗi xảy ra: {str(e)}")
            full_response = "Xin lỗi, tôi gặp sự cố khi xử lý câu hỏi của bạn. Vui lòng thử lại."
        
    # Add response to history and memory
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.total_messages = len(st.session_state.messages)
    
    # Update memory with the conversation
    memory.save_context({"input": prompt}, {"answer": full_response})

# Quick question suggestions
if not st.session_state.messages or len(st.session_state.messages) < 2:
    st.markdown("### 💡 Câu hỏi gợi ý:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Tìm định nghĩa cơ bản", use_container_width=True):
            st.session_state.suggested_question = "Hãy giải thích các định nghĩa cơ bản trong tài liệu"
            st.rerun()
            
        if st.button("📊 So sánh các khái niệm", use_container_width=True):
            st.session_state.suggested_question = "So sánh các khái niệm chính trong tài liệu"
            st.rerun()
    
    with col2:
        if st.button("🧮 Giải thích thuật toán", use_container_width=True):
            st.session_state.suggested_question = "Giải thích các thuật toán được đề cập trong tài liệu"
            st.rerun()
            
        if st.button("📝 Tóm tắt nội dung", use_container_width=True):
            st.session_state.suggested_question = "Tóm tắt những điểm chính trong tài liệu"
            st.rerun()
