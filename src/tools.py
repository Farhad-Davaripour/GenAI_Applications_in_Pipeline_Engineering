import nest_asyncio
nest_asyncio.apply()
from copy import deepcopy
from llama_index.core.schema import TextNode
import base64
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet, InvalidToken
import os


def get_nodes(docs):
    """Split docs into nodes, by separator."""
    nodes = []
    for doc in docs:
        doc_chunks = doc.text.split("\n---\n")
        for doc_chunk in doc_chunks:
            node = TextNode(
                text=doc_chunk,
                metadata=deepcopy(doc.metadata),
            )
            nodes.append(node)

    return nodes

def calculate_circumferential_stress(P, D, t_n):
    """
    Calculate the circumferential (hoop) stress in a pipeline based on CSA Z662 standard (Section 4.8.3).
    
    Args:
    P (float): Design pressure of the pipeline (in MPa)
    D (float): Outside diameter of the pipe (in mm)
    t_n (float): Pipe nominal wall thickness, less allowances (in mm)
    
    Returns:
    float: Circumferential (hoop) stress (in MPa)
    """
    S_h = (P * D) / (2 * t_n)
    return S_h


def generate_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=390000,
        backend=default_backend()
    )
    key = kdf.derive(password.encode())
    return key

def encrypt_data(data: bytes, password: str) -> bytes:
    salt = os.urandom(16)  # Generate new salt for each encryption
    key = generate_key(password, salt)
    cipher = Fernet(base64.urlsafe_b64encode(key))
    encrypted_data = cipher.encrypt(data)
    return salt + encrypted_data  # Prepend salt to encrypted data for storage

def decrypt_data(encrypted_data: bytes, password: str) -> bytes:
    salt = encrypted_data[:16]  # Extract the salt
    key = generate_key(password, salt)
    cipher = Fernet(base64.urlsafe_b64encode(key))
    try:
        decrypted_data = cipher.decrypt(encrypted_data[16:])  # Remove the salt
        return decrypted_data
    except InvalidToken:
        raise ValueError("Incorrect password or corrupted data")