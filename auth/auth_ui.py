"""
Authentication UI Components
Clean, professional login/register interface
"""

import streamlit as st
from auth.auth_manager import AuthManager
from typing import Optional

class AuthUI:
    def __init__(self, auth_manager: AuthManager):
        self.auth = auth_manager
    
    def render_auth_page(self) -> bool:
        """Render authentication page, returns True if authenticated"""
        if self.auth.is_authenticated():
            return True
        
        # Clean, minimal styling
        st.markdown("""
        <style>
        .auth-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .auth-title {
            text-align: center;
            color: #ffffff;
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 2rem;
        }
        .auth-form {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: white;
            padding: 0.75rem;
        }
        .stTextInput > div > div > input:focus {
            border-color: #4ECDC4;
            box-shadow: 0 0 0 2px rgba(78, 205, 196, 0.2);
        }
        .auth-button {
            background: linear-gradient(135deg, #4ECDC4, #44A08D);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .auth-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(78, 205, 196, 0.3);
        }
        .auth-link {
            text-align: center;
            color: #4ECDC4;
            cursor: pointer;
            text-decoration: underline;
            margin-top: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown("""
        <div style="text-align: center; margin-bottom: 3rem;">
            <h1 style="color: #4ECDC4; font-size: 2.5rem; margin-bottom: 0.5rem;">🧠 AI Trợ Giảng</h1>
            <p style="color: rgba(255,255,255,0.7); font-size: 1.1rem;">Đăng nhập để truy cập chatbot AI</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Auth form container
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Toggle between login and register
                if "auth_mode" not in st.session_state:
                    st.session_state.auth_mode = "login"
                
                if st.session_state.auth_mode == "login":
                    self._render_login_form()
                else:
                    self._render_register_form()
        
        return False
    
    def _render_login_form(self):
        """Render login form"""
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="auth-title">Đăng Nhập</h2>', unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("📧 Email", placeholder="your@email.com")
            password = st.text_input("🔒 Mật khẩu", type="password", placeholder="Nhập mật khẩu")
            
            col1, col2 = st.columns(2)
            with col1:
                login_btn = st.form_submit_button("Đăng Nhập", use_container_width=True)
            with col2:
                forgot_btn = st.form_submit_button("Quên mật khẩu?", use_container_width=True)
            
            if login_btn and email and password:
                success, message = self.auth.sign_in(email, password)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            
            if forgot_btn and email:
                success, message = self.auth.reset_password(email)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # Switch to register
        if st.button("Chưa có tài khoản? Đăng ký ngay", use_container_width=True):
            st.session_state.auth_mode = "register"
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_register_form(self):
        """Render registration form"""
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="auth-title">Đăng Ký</h2>', unsafe_allow_html=True)
        
        with st.form("register_form", clear_on_submit=False):
            full_name = st.text_input("👤 Họ và tên", placeholder="Nguyễn Văn A")
            email = st.text_input("📧 Email", placeholder="your@email.com")
            password = st.text_input("🔒 Mật khẩu", type="password", placeholder="Ít nhất 6 ký tự")
            confirm_password = st.text_input("🔒 Xác nhận mật khẩu", type="password", placeholder="Nhập lại mật khẩu")
            
            register_btn = st.form_submit_button("Đăng Ký", use_container_width=True)
            
            if register_btn:
                if not all([full_name, email, password, confirm_password]):
                    st.error("Vui lòng điền đầy đủ thông tin")
                elif password != confirm_password:
                    st.error("Mật khẩu xác nhận không khớp")
                elif len(password) < 6:
                    st.error("Mật khẩu phải có ít nhất 6 ký tự")
                else:
                    success, message = self.auth.sign_up(email, password, full_name)
                    if success:
                        st.success(message)
                        st.session_state.auth_mode = "login"
                        st.rerun()
                    else:
                        st.error(message)
        
        # Switch to login
        if st.button("Đã có tài khoản? Đăng nhập", use_container_width=True):
            st.session_state.auth_mode = "login"
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_user_profile(self):
        """Render user profile in sidebar"""
        if not self.auth.is_authenticated():
            return
        
        user = self.auth.get_current_user()
        stats = self.auth.get_user_stats()
        is_admin = self.auth.is_admin()
        
        st.markdown("### 👤 Thông tin tài khoản")
        
        # Admin badge
        if is_admin:
            st.markdown("🔑 **ADMIN** - Quản trị viên")
        
        # User info
        st.markdown(f"""
        **Tên:** {stats.get('full_name', 'Chưa cập nhật')}  
        **Email:** {stats.get('email', '')}  
        **Tin nhắn:** {stats.get('total_messages', 0)}  
        **Tài liệu:** {stats.get('uploaded_files', 0)}
        """)
        
        st.markdown("---")
        
        # Profile update
        with st.expander("✏️ Cập nhật thông tin"):
            new_name = st.text_input("Họ và tên mới", value=stats.get('full_name', ''))
            if st.button("Cập nhật", use_container_width=True):
                if new_name.strip():
                    success, message = self.auth.update_profile(new_name.strip())
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
        
        # Logout button
        if st.button("🚪 Đăng xuất", use_container_width=True, type="secondary"):
            success, message = self.auth.sign_out()
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
