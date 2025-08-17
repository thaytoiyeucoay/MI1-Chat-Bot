"""
Authentication Manager for AI Trợ Giảng Toán Tin
Handles Supabase authentication with clean, professional interface
"""

import streamlit as st
from supabase import Client
from typing import Optional, Dict, Any
import hashlib
import time
import os

class AuthManager:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.session_key = "auth_session"
        self.user_key = "auth_user"
        # Admin credentials - change these to your desired admin email/password
        self.admin_email = os.getenv("ADMIN_EMAIL", "admin@toanhoc.edu.vn")
        self.admin_password = os.getenv("ADMIN_PASSWORD", "AdminToanhoc2024!")
    
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated"""
        return self.session_key in st.session_state and st.session_state[self.session_key] is not None
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current authenticated user info"""
        if self.is_authenticated():
            return st.session_state.get(self.user_key)
        return None
    
    def is_admin(self) -> bool:
        """Check if current user is admin"""
        user = self.get_current_user()
        if user:
            return user.get("email") == self.admin_email
        return False
    
    def can_upload_documents(self) -> bool:
        """Check if current user can upload documents (admin only)"""
        return self.is_admin()
    
    def sign_up(self, email: str, password: str, full_name: str) -> tuple[bool, str]:
        """Register new user"""
        try:
            # Create user with Supabase Auth
            response = self.supabase.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": {
                        "full_name": full_name
                    }
                }
            })
            
            if response.user:
                return True, "Đăng ký thành công! Vui lòng kiểm tra email để xác thực tài khoản."
            else:
                return False, "Đăng ký thất bại. Vui lòng thử lại."
                
        except Exception as e:
            error_msg = str(e)
            if "already registered" in error_msg.lower():
                return False, "Email này đã được đăng ký. Vui lòng sử dụng email khác."
            elif "invalid email" in error_msg.lower():
                return False, "Email không hợp lệ."
            elif "password" in error_msg.lower():
                return False, "Mật khẩu phải có ít nhất 6 ký tự."
            else:
                return False, f"Lỗi đăng ký: {error_msg}"
    
    def sign_in(self, email: str, password: str) -> tuple[bool, str]:
        """Sign in user"""
        try:
            response = self.supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if response.user and response.session:
                # Store session in Streamlit state
                st.session_state[self.session_key] = response.session
                st.session_state[self.user_key] = {
                    "id": response.user.id,
                    "email": response.user.email,
                    "full_name": response.user.user_metadata.get("full_name", ""),
                    "created_at": response.user.created_at
                }
                return True, "Đăng nhập thành công!"
            else:
                return False, "Đăng nhập thất bại."
                
        except Exception as e:
            error_msg = str(e)
            if "invalid credentials" in error_msg.lower() or "invalid login" in error_msg.lower():
                return False, "Email hoặc mật khẩu không chính xác."
            elif "email not confirmed" in error_msg.lower():
                return False, "Vui lòng xác thực email trước khi đăng nhập."
            else:
                return False, f"Lỗi đăng nhập: {error_msg}"
    
    def sign_out(self) -> tuple[bool, str]:
        """Sign out current user"""
        try:
            self.supabase.auth.sign_out()
            
            # Clear session state
            if self.session_key in st.session_state:
                del st.session_state[self.session_key]
            if self.user_key in st.session_state:
                del st.session_state[self.user_key]
            
            # Clear other app-specific session data
            keys_to_clear = ["messages", "uploaded_files_count", "total_messages"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            return True, "Đăng xuất thành công!"
            
        except Exception as e:
            return False, f"Lỗi đăng xuất: {str(e)}"
    
    def reset_password(self, email: str) -> tuple[bool, str]:
        """Send password reset email"""
        try:
            self.supabase.auth.reset_password_email(email)
            return True, "Email đặt lại mật khẩu đã được gửi!"
            
        except Exception as e:
            return False, f"Lỗi gửi email: {str(e)}"
    
    def update_profile(self, full_name: str) -> tuple[bool, str]:
        """Update user profile"""
        try:
            if not self.is_authenticated():
                return False, "Bạn cần đăng nhập để cập nhật thông tin."
            
            response = self.supabase.auth.update_user({
                "data": {"full_name": full_name}
            })
            
            if response.user:
                # Update session state
                if self.user_key in st.session_state:
                    st.session_state[self.user_key]["full_name"] = full_name
                return True, "Cập nhật thông tin thành công!"
            else:
                return False, "Cập nhật thất bại."
                
        except Exception as e:
            return False, f"Lỗi cập nhật: {str(e)}"
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics"""
        if not self.is_authenticated():
            return {}
        
        user = self.get_current_user()
        return {
            "email": user.get("email", ""),
            "full_name": user.get("full_name", ""),
            "member_since": user.get("created_at", ""),
            "total_messages": st.session_state.get("total_messages", 0),
            "uploaded_files": st.session_state.get("uploaded_files_count", 0)
        }
