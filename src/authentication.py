"""
Authentication module for Q-Rescue application.
Implements secure user registration, bcrypt password hashing, login verification,
and Streamlit session state management.
"""

import bcrypt
import streamlit as st
import re
from typing import Optional, Dict, Any, Tuple
from src.database import get_connection

def hash_password(password: str) -> str:
    """Hash password using bcrypt with salt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify plaintext password against bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False

def validate_username(username: str) -> Tuple[bool, str]:
    """Validate username rules: 3-30 chars, alphanumeric + underscores."""
    if not username or len(username.strip()) < 3:
        return False, "Username must be at least 3 characters long."
    if not re.match(r"^[a-zA-Z0-9_]{3,30}$", username.strip()):
        return False, "Username may only contain letters, numbers, and underscores."
    return True, ""

def validate_password(password: str) -> Tuple[bool, str]:
    """Validate password complexity: at least 6 characters."""
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters long."
    return True, ""

def register_user(username: str, password: str, full_name: str = "", 
                  email: str = "", role: str = "Disaster Responder") -> Tuple[bool, str]:
    """Register a new user in SQLite with bcrypt hashed password."""
    valid_u, msg_u = validate_username(username)
    if not valid_u:
        return False, msg_u
        
    valid_p, msg_p = validate_password(password)
    if not valid_p:
        return False, msg_p
        
    pw_hash = hash_password(password)
    username_clean = username.strip().lower()
    
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (username, password_hash, email, full_name, role)
                VALUES (?, ?, ?, ?, ?);
            """, (username_clean, pw_hash, email.strip(), full_name.strip(), role))
            conn.commit()
            return True, "Registration successful! You may now log in."
    except Exception as e:
        if "UNIQUE constraint failed" in str(e):
            return False, f"Username '{username_clean}' is already registered. Please choose another."
        return False, f"Registration error: {str(e)}"

def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Verify user credentials and update last_login timestamp."""
    username_clean = username.strip().lower()
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, username, password_hash, email, full_name, role
                FROM users WHERE username = ?;
            """, (username_clean,))
            user = cursor.fetchone()
            
            if not user:
                return False, None, "Invalid username or password."
                
            if verify_password(password, user["password_hash"]):
                cursor.execute("""
                    UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?;
                """, (user["user_id"],))
                conn.commit()
                user_dict = {
                    "user_id": user["user_id"],
                    "username": user["username"],
                    "full_name": user["full_name"],
                    "email": user["email"],
                    "role": user["role"]
                }
                return True, user_dict, "Login successful."
            else:
                return False, None, "Invalid username or password."
    except Exception as e:
        return False, None, f"Database error during authentication: {str(e)}"

def init_auth_state() -> None:
    """Initialize session state keys for user authentication."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user" not in st.session_state:
        st.session_state["user"] = None

def login_user_session(user_dict: Dict[str, Any]) -> None:
    """Set authenticated user session in Streamlit."""
    st.session_state["authenticated"] = True
    st.session_state["user"] = user_dict

def logout_user_session() -> None:
    """Clear user session."""
    st.session_state["authenticated"] = False
    st.session_state["user"] = None
    st.rerun()

def get_current_user() -> Optional[Dict[str, Any]]:
    """Return currently authenticated user from session state."""
    init_auth_state()
    return st.session_state.get("user")

def render_auth_sidebar() -> None:
    """Render user profile and logout button in the sidebar if authenticated."""
    init_auth_state()
    if st.session_state.get("authenticated") and st.session_state.get("user"):
        user = st.session_state["user"]
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"👤 **User:** `{user.get('username')}`")
            st.caption(f"Role: {user.get('role')} | {user.get('full_name', '')}")
            if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
                logout_user_session()

def require_auth() -> bool:
    """Helper for protected pages. Returns True if authenticated, else renders login notice."""
    init_auth_state()
    if not st.session_state.get("authenticated"):
        st.warning("🔒 **Authentication Required:** Please log in or register on the Home / Login page to access this emergency decision-support feature.")
        st.info("💡 You can register a new account on the Home page or use the demo credentials provided.")
        return False
    return True
