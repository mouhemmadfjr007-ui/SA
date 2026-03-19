"""
Sentiment Arabia - Authentication & Authorization
JWT + OAuth2 مع عزل بيانات الشركات
"""
import os
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = os.getenv("JWT_SECRET", "sentiment-arabia-secret-key-2024-ultra-secure")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

bearer_scheme = HTTPBearer(auto_error=False)

def create_token(data: Dict, expires_hours: int = ACCESS_TOKEN_EXPIRE_HOURS) -> str:
    payload = data.copy()
    expire = datetime.utcnow() + timedelta(hours=expires_hours)
    payload.update({"exp": expire, "iat": datetime.utcnow()})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> Optional[Dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None

def get_current_company(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
) -> Dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="يجب تسجيل الدخول أولاً")
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="رمز التحقق غير صالح أو منتهي")
    return payload

def get_current_company_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
) -> Optional[Dict]:
    if not credentials:
        return None
    return decode_token(credentials.credentials)

def check_company_permission(company_id: str, current_user: Dict) -> bool:
    return current_user.get("company_id") == company_id or current_user.get("role") == "superadmin"
