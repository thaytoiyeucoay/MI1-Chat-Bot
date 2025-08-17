"""
Demo script để test và so sánh các chiến lược Semantic Chunking
"""

import streamlit as st
from semantic_chunking import ChunkingManager
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_sample_documents():
    """Tạo các tài liệu mẫu để test"""
    
    # Tài liệu toán học
    math_content = """
    Định lý 1: Định lý Pythagoras
    Trong một tam giác vuông, bình phương cạnh huyền bằng tổng bình phương hai cạnh góc vuông.
    Công thức: a² + b² = c²
    
    Chứng minh:
    Xét tam giác ABC vuông tại A, với BC là cạnh huyền.
    Vẽ đường cao AH từ A xuống BC.
    Ta có: AB² = BH × BC và AC² = CH × BC
    Do đó: AB² + AC² = BH × BC + CH × BC = BC × (BH + CH) = BC²
    
    Ví dụ 1: Cho tam giác vuông có hai cạnh góc vuông là 3 và 4.
    Tính cạnh huyền: c = √(3² + 4²) = √(9 + 16) = √25 = 5
    
    Định lý 2: Định lý cosin
    Trong tam giác ABC bất kỳ: a² = b² + c² - 2bc×cos(A)
    """
    
    # Tài liệu có cấu trúc
    structured_content = """
    Chương 1: Giới thiệu về Cấu trúc Dữ liệu
    
    1.1 Định nghĩa
    Cấu trúc dữ liệu là cách tổ chức và lưu trữ dữ liệu trong máy tính.
    
    1.2 Phân loại
    a) Cấu trúc tuyến tính:
       - Mảng (Array)
       - Danh sách liên kết (Linked List)
       - Ngăn xếp (Stack)
       - Hàng đợi (Queue)
    
    b) Cấu trúc phi tuyến:
       - Cây (Tree)
       - Đồ thị (Graph)
    
    1.3 Ứng dụng
    • Tối ưu hóa thuật toán
    • Quản lý bộ nhớ hiệu quả
    • Tăng tốc độ truy xuất dữ liệu
    
    Chương 2: Mảng và Danh sách
    
    2.1 Mảng
    Mảng là tập hợp các phần tử cùng kiểu dữ liệu.
    Ưu điểm: Truy xuất nhanh theo chỉ số
    Nhược điểm: Kích thước cố định
    """
    
    # Tài liệu tự nhiên
    narrative_content = """
    Lịch sử phát triển của máy tính điện tử bắt đầu từ những năm 1940. 
    Máy tính đầu tiên ENIAC được xây dựng tại Đại học Pennsylvania vào năm 1946.
    Nó nặng tới 30 tấn và chiếm diện tích 167 mét vuông.
    
    Trong những thập kỷ tiếp theo, công nghệ máy tính đã có những bước tiến vượt bậc.
    Việc phát minh ra transistor vào năm 1947 đã mở ra kỷ nguyên mới.
    Sau đó là sự ra đời của vi xử lý vào những năm 1970.
    
    Ngày nay, máy tính đã trở thành một phần không thể thiếu trong cuộc sống.
    Từ điện thoại thông minh đến siêu máy tính, chúng đều dựa trên những nguyên lý cơ bản giống nhau.
    Trí tuệ nhân tạo và học máy đang định hình tương lai của công nghệ.
    """
    
    return [
        Document(page_content=math_content, metadata={"type": "mathematical", "subject": "geometry"}),
        Document(page_content=structured_content, metadata={"type": "structured", "subject": "data_structures"}),
        Document(page_content=narrative_content, metadata={"type": "narrative", "subject": "computer_history"})
    ]

def run_chunking_comparison():
    """Chạy so sánh các chiến lược chunking"""
    
    st.title("🧠 Demo Semantic Chunking")
    st.markdown("---")
    
    # Initialize components
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            st.error("Vui lòng thiết lập GOOGLE_API_KEY trong file .env")
            return
            
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        chunking_manager = ChunkingManager(embeddings)
        
        st.success("✅ Đã khởi tạo thành công Semantic Chunking Manager")
        
    except Exception as e:
        st.error(f"Lỗi khởi tạo: {e}")
        return
    
    # Create sample documents
    sample_docs = create_sample_documents()
    
    st.markdown("## 📄 Tài liệu mẫu")
    for i, doc in enumerate(sample_docs):
        with st.expander(f"Tài liệu {i+1}: {doc.metadata['subject']}"):
            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
    
    st.markdown("---")
    st.markdown("## 🔬 So sánh các chiến lược Chunking")
    
    # Get available strategies
    strategies = chunking_manager.get_available_strategies()
    
    results = {}
    
    # Test each strategy
    for strategy_name, strategy_desc in strategies.items():
        st.markdown(f"### {strategy_name.title()}")
        st.info(f"📝 {strategy_desc}")
        
        with st.spinner(f"Đang test {strategy_name}..."):
            start_time = time.time()
            
            try:
                chunks, stats = chunking_manager.process_documents(sample_docs, strategy_name)
                processing_time = time.time() - start_time
                
                results[strategy_name] = {
                    'chunks': chunks,
                    'stats': stats,
                    'processing_time': processing_time
                }
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Tổng chunks", stats.get('total_chunks', 0))
                with col2:
                    st.metric("Độ dài TB", f"{stats.get('avg_chunk_length', 0):.0f}")
                with col3:
                    st.metric("Thời gian", f"{processing_time:.2f}s")
                with col4:
                    st.metric("Hiệu quả", f"{stats.get('total_chunks', 0)/processing_time:.1f} chunks/s")
                
                # Show chunk types
                chunk_types = stats.get('chunk_types', {})
                if chunk_types:
                    st.write("📊 **Phân bố loại chunks:**")
                    for chunk_type, count in chunk_types.items():
                        st.write(f"• {chunk_type}: {count}")
                
                # Show sample chunks
                with st.expander("👀 Xem mẫu chunks"):
                    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                        st.write(f"**Chunk {i+1}** ({chunk.metadata.get('chunk_type', 'unknown')}):")
                        st.text(chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content)
                        st.write("---")
                
            except Exception as e:
                st.error(f"Lỗi khi test {strategy_name}: {e}")
        
        st.markdown("---")
    
    # Summary comparison
    if results:
        st.markdown("## 📊 Tổng kết So sánh")
        
        comparison_data = []
        for strategy, result in results.items():
            stats = result['stats']
            comparison_data.append({
                'Chiến lược': strategy.title(),
                'Tổng chunks': stats.get('total_chunks', 0),
                'Độ dài TB': f"{stats.get('avg_chunk_length', 0):.0f}",
                'Thời gian (s)': f"{result['processing_time']:.2f}",
                'Hiệu quả': f"{stats.get('total_chunks', 0)/result['processing_time']:.1f}"
            })
        
        st.dataframe(comparison_data)
        
        # Recommendations
        st.markdown("## 💡 Khuyến nghị")
        
        best_adaptive = max(results.items(), key=lambda x: x[1]['stats'].get('total_chunks', 0) if x[0] == 'adaptive' else 0)
        fastest = min(results.items(), key=lambda x: x[1]['processing_time'])
        most_chunks = max(results.items(), key=lambda x: x[1]['stats'].get('total_chunks', 0))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"🎯 **Tốt nhất tổng thể**: {best_adaptive[0].title()}")
        with col2:
            st.info(f"⚡ **Nhanh nhất**: {fastest[0].title()}")
        with col3:
            st.warning(f"📈 **Nhiều chunks nhất**: {most_chunks[0].title()}")

if __name__ == "__main__":
    run_chunking_comparison()
