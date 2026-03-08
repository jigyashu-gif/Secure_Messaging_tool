"""Secure end-to-end messenger using CustomTkinter, RSA, and AES-GCM.

This module provides a small GUI application that can act as either a server
or a client. Each peer:

- Generates its own RSA key pair.
- Exchanges RSA public keys over a TCP socket.
- Establishes a shared AES-256 key (encrypted with RSA).
- Uses AES-GCM for authenticated encryption of all chat messages.

The GUI stays responsive because all blocking network operations run in
background threads. All UI updates are marshalled back to the Tk main
thread using ``after(...)``.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import socket
import struct
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import customtkinter as ctk
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ===================== CONFIGURATION / LOGGING =====================

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_HOST_SERVER = "0.0.0.0"
DEFAULT_HOST_CLIENT = "127.0.0.1"
DEFAULT_PORT = 5000
AES_NONCE_SIZE = 12  # bytes, 96 bits – recommended for AES-GCM

IDENTITY_PRIV_FILE = BASE_DIR / "id_rsa_private.pem"
IDENTITY_PUB_FILE = BASE_DIR / "id_rsa_public.pem"
FRIENDS_FILE = BASE_DIR / "friends.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("secure_chat")


# ===================== CRYPTOGRAPHY HELPERS =====================

def generate_rsa_keypair() -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
    """
    Generate an RSA private/public key pair.

    RSA is used only for:
      - Exchanging the symmetric AES key securely.
      - Not for encrypting every chat message (too slow/inefficient).
    """
    private_key: rsa.RSAPrivateKey = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    public_key: rsa.RSAPublicKey = private_key.public_key()
    return private_key, public_key


def load_or_generate_rsa_identity() -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
    """Load a long-lived RSA identity key pair from disk, or create one.

    This gives each installation a stable asymmetric identity which is then
    used to derive per-session AES keys with peers.
    """
    if IDENTITY_PRIV_FILE.exists() and IDENTITY_PUB_FILE.exists():
        LOGGER.info("Loading existing RSA identity from disk.")
        private_bytes = IDENTITY_PRIV_FILE.read_bytes()
        public_bytes = IDENTITY_PUB_FILE.read_bytes()

        private_key = serialization.load_pem_private_key(
            private_bytes, password=None
        )
        public_key = serialization.load_pem_public_key(public_bytes)  # type: ignore[arg-type]
        return private_key, public_key  # type: ignore[return-value]

    LOGGER.info("Generating new RSA identity and writing to disk.")
    private_key, public_key = generate_rsa_keypair()

    priv_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_bytes = serialize_public_key(public_key)

    IDENTITY_PRIV_FILE.write_bytes(priv_bytes)
    IDENTITY_PUB_FILE.write_bytes(pub_bytes)

    return private_key, public_key


def serialize_public_key(public_key: rsa.RSAPublicKey) -> bytes:
    """
    Convert an RSA public key to PEM bytes to send over the network.
    """
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def deserialize_public_key(pem_bytes: bytes) -> rsa.RSAPublicKey:
    """
    Convert PEM bytes back to an RSA public key object.
    """
    return serialization.load_pem_public_key(pem_bytes)  # type: ignore[return-value]


def rsa_encrypt(public_key: rsa.RSAPublicKey, plaintext_bytes: bytes) -> bytes:
    """
    Encrypt data with an RSA public key using OAEP.

    Used to encrypt the randomly generated AES key so that
    only the holder of the matching RSA private key can recover it.
    """
    ciphertext: bytes = public_key.encrypt(
        plaintext_bytes,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return ciphertext


def rsa_decrypt(private_key: rsa.RSAPrivateKey, ciphertext_bytes: bytes) -> bytes:
    """
    Decrypt data encrypted with rsa_encrypt using the RSA private key.
    """
    plaintext: bytes = private_key.decrypt(
        ciphertext_bytes,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return plaintext


def generate_aes_key() -> bytes:
    """
    Generate a random 256-bit AES key (32 bytes) for symmetric encryption.
    """
    return AESGCM.generate_key(bit_length=256)


def aes_encrypt(aes_key: bytes, plaintext_bytes: bytes) -> Tuple[bytes, bytes]:
    """
    Encrypt a message with AES-GCM (AEAD mode).

    - A fresh random nonce is generated for each message.
    - AES-GCM provides both confidentiality and integrity.
    """
    aesgcm = AESGCM(aes_key)
    nonce = os.urandom(AES_NONCE_SIZE)
    ciphertext = aesgcm.encrypt(nonce, plaintext_bytes, associated_data=None)
    return nonce, ciphertext


def aes_decrypt(aes_key: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
    """
    Decrypt a message encrypted with aes_encrypt using AES-GCM.

    If the ciphertext has been modified, this will raise an exception,
    protecting against tampering.
    """
    aesgcm = AESGCM(aes_key)
    plaintext: bytes = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    return plaintext


# ===================== SOCKET MESSAGE FRAMING =====================

def send_message(sock: socket.socket, obj: Dict[str, Any]) -> None:
    """
    Send a JSON-serializable object over a socket with length-prefix framing.

    Format:
      [4-byte big-endian length][JSON bytes]
    """
    data = json.dumps(obj).encode("utf-8")
    length_prefix = struct.pack("!I", len(data))
    sock.sendall(length_prefix + data)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """
    Receive exactly n bytes from a socket. Raise if connection closes early.
    """
    chunks: list[bytes] = []
    bytes_recd = 0
    while bytes_recd < n:
        chunk = sock.recv(n - bytes_recd)
        if chunk == b"":
            raise ConnectionError("Socket connection closed")
        chunks.append(chunk)
        bytes_recd += len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> Dict[str, Any]:
    """
    Receive a single length-prefixed JSON object from a socket.
    """
    # First read the 4-byte length prefix
    length_prefix = recv_exact(sock, 4)
    (msg_len,) = struct.unpack("!I", length_prefix)
    data = recv_exact(sock, msg_len)
    obj: Dict[str, Any] = json.loads(data.decode("utf-8"))
    return obj


# ===================== GUI APPLICATION =====================

class SecureChatApp(ctk.CTk):
    """Main GUI application for the secure chat."""

    def __init__(self) -> None:
        super().__init__()

        # ---- Window setup ----
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Secure E2E Messenger (RSA + AES)")
        self.geometry("800x600")

        # ---- Crypto state ----
        # Long-lived asymmetric identity for this installation
        self.rsa_private_key, self.rsa_public_key = load_or_generate_rsa_identity()
        self.peer_public_key: Optional[rsa.RSAPublicKey] = None
        self.aes_key: Optional[bytes] = None  # shared symmetric key (once established)

        # ---- Friends / address book ----
        # Each friend: {"name": str, "host": str, "port": int}
        self.friends: List[Dict[str, Any]] = []
        self.friend_var = ctk.StringVar(value="")

        # ---- Networking state ----
        self.role = ctk.StringVar(value="server")  # "server" or "client"
        self.sock: Optional[socket.socket] = None
        self.server_socket: Optional[socket.socket] = None
        self.connected: bool = False

        # Thread synchronization / shutdown
        self.recv_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # ---- Build GUI ----
        self._build_gui()

        # On close, cleanup sockets/threads cleanly
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ----------------- GUI LAYOUT -----------------

    def _build_gui(self) -> None:
        # Top frame: connection controls
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(side="top", fill="x", padx=10, pady=10)

        # Role selection
        role_label = ctk.CTkLabel(top_frame, text="Role:")
        role_label.grid(row=0, column=0, padx=5)

        server_radio = ctk.CTkRadioButton(
            top_frame,
            text="Server",
            variable=self.role,
            value="server",
            command=self._on_role_change,
        )
        server_radio.grid(row=0, column=1, padx=5)

        client_radio = ctk.CTkRadioButton(
            top_frame,
            text="Client",
            variable=self.role,
            value="client",
            command=self._on_role_change,
        )
        client_radio.grid(row=0, column=2, padx=5)

        # Host
        host_label = ctk.CTkLabel(top_frame, text="Host:")
        host_label.grid(row=0, column=3, padx=(15, 5))

        self.host_entry = ctk.CTkEntry(top_frame, width=160)
        self.host_entry.grid(row=0, column=4, padx=5)
        self.host_entry.insert(0, DEFAULT_HOST_SERVER)  # for server; client will overwrite

        # Port
        port_label = ctk.CTkLabel(top_frame, text="Port:")
        port_label.grid(row=0, column=5, padx=(15, 5))

        self.port_entry = ctk.CTkEntry(top_frame, width=80)
        self.port_entry.grid(row=0, column=6, padx=5)
        self.port_entry.insert(0, str(DEFAULT_PORT))

        # Connect button
        self.connect_button = ctk.CTkButton(
            top_frame, text="Connect", command=self.on_connect_click
        )
        self.connect_button.grid(row=0, column=7, padx=(15, 5))

        # Status label
        self.status_label = ctk.CTkLabel(
            top_frame, text="Not connected", text_color="red"
        )
        self.status_label.grid(row=0, column=8, padx=10)

        # Second row: simple "friends" / address-book controls
        friend_label = ctk.CTkLabel(top_frame, text="Friend:")
        friend_label.grid(row=1, column=0, padx=5, pady=(5, 0))

        self.friend_name_entry = ctk.CTkEntry(top_frame, width=160, placeholder_text="Nickname")
        self.friend_name_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=(5, 0), sticky="we")

        self.friend_menu = ctk.CTkOptionMenu(
            top_frame,
            variable=self.friend_var,
            values=[],
            command=self._on_friend_selected,
        )
        self.friend_menu.grid(row=1, column=3, columnspan=2, padx=5, pady=(5, 0), sticky="we")

        save_friend_button = ctk.CTkButton(
            top_frame,
            text="Save / Update Friend",
            command=self.on_save_friend_click,
            width=140,
        )
        save_friend_button.grid(row=1, column=5, columnspan=2, padx=5, pady=(5, 0))

        # Middle frame: chat history
        mid_frame = ctk.CTkFrame(self)
        mid_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 10))

        history_label = ctk.CTkLabel(mid_frame, text="Conversation:")
        history_label.pack(anchor="w", padx=5, pady=(5, 0))

        self.chat_history = ctk.CTkTextbox(mid_frame, wrap="word")
        self.chat_history.pack(fill="both", expand=True, padx=5, pady=5)
        self.chat_history.configure(state="disabled")

        # Bottom frame: message entry and send button
        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.pack(side="bottom", fill="x", padx=10, pady=(0, 10))

        msg_label = ctk.CTkLabel(bottom_frame, text="Message:")
        msg_label.grid(row=0, column=0, padx=5, pady=5)

        self.message_entry = ctk.CTkEntry(bottom_frame, width=550)
        self.message_entry.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        self.message_entry.bind("<Return>", lambda event: self.on_send_click())

        self.send_button = ctk.CTkButton(
            bottom_frame, text="Send", command=self.on_send_click, state="disabled"
        )
        self.send_button.grid(row=0, column=2, padx=5, pady=5)

        bottom_frame.grid_columnconfigure(1, weight=1)

        # Load any existing friends from disk and populate the menu.
        self._load_friends()
        self._refresh_friend_menu()

    # ----------------- GUI UTILITIES -----------------

    def append_chat(self, text: str) -> None:
        """
        Thread-safe way to append a line to the chat history box.
        """
        def _append():
            self.chat_history.configure(state="normal")
            self.chat_history.insert("end", text + "\n")
            self.chat_history.see("end")
            self.chat_history.configure(state="disabled")

        # Ensure this is executed in the Tkinter main thread
        self.after(0, _append)

    def set_status(self, text: str, color: str = "white") -> None:
        def _set():
            self.status_label.configure(text=text, text_color=color)
        self.after(0, _set)

    def _on_role_change(self) -> None:
        """
        When the user switches between Server and Client,
        adjust default host field for convenience.
        """
        role = self.role.get()
        if role == "server":
            if not self.connected:
                self.host_entry.delete(0, "end")
                # Bind to all interfaces by default for LAN use
                self.host_entry.insert(0, DEFAULT_HOST_SERVER)
        else:
            if not self.connected:
                self.host_entry.delete(0, "end")
                # For localhost use, user can keep 127.0.0.1 or enter LAN IP.
                self.host_entry.insert(0, DEFAULT_HOST_CLIENT)

    # ----------------- FRIEND MANAGEMENT -----------------

    def _load_friends(self) -> None:
        """Load saved friends from JSON file, if present."""
        if not FRIENDS_FILE.exists():
            return
        try:
            data = json.loads(FRIENDS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self.friends = [
                    f
                    for f in data
                    if isinstance(f, dict)
                    and "name" in f
                    and "host" in f
                    and "port" in f
                ]
        except Exception as exc:
            LOGGER.warning("Failed to load friends.json: %s", exc)

    def _save_friends(self) -> None:
        """Persist friends list to JSON file."""
        try:
            FRIENDS_FILE.write_text(
                json.dumps(self.friends, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            LOGGER.warning("Failed to save friends.json: %s", exc)

    def _refresh_friend_menu(self) -> None:
        """Refresh the dropdown of known friends."""
        names = [f["name"] for f in self.friends]
        if not names:
            # Ensure menu still works when empty
            self.friend_menu.configure(values=[""])
            self.friend_var.set("")
            return

        self.friend_menu.configure(values=names)
        if self.friend_var.get() not in names:
            self.friend_var.set(names[0])

    def _on_friend_selected(self, name: str) -> None:
        """Populate host/port fields when a friend is selected."""
        for friend in self.friends:
            if friend.get("name") == name:
                self.host_entry.delete(0, "end")
                self.host_entry.insert(0, str(friend.get("host", "")))

                self.port_entry.delete(0, "end")
                self.port_entry.insert(0, str(friend.get("port", DEFAULT_PORT)))

                self.friend_name_entry.delete(0, "end")
                self.friend_name_entry.insert(0, name)
                break

    def on_save_friend_click(self) -> None:
        """Save or update a friend entry using current host/port."""
        name = self.friend_name_entry.get().strip()
        host = self.host_entry.get().strip()
        port_str = self.port_entry.get().strip()

        if not name or not host or not port_str:
            self.append_chat(
                "[System] Friend name, host, and port are required to save a friend."
            )
            return

        try:
            port = int(port_str)
        except ValueError:
            self.append_chat("[Error] Port must be an integer to save a friend.")
            return

        existing = next((f for f in self.friends if f.get("name") == name), None)
        if existing:
            existing["host"] = host
            existing["port"] = port
        else:
            self.friends.append({"name": name, "host": host, "port": port})

        self._save_friends()
        self._refresh_friend_menu()
        self.append_chat(f"[System] Friend '{name}' saved with {host}:{port}.")

    # ----------------- CONNECTION HANDLERS -----------------

    def on_connect_click(self) -> None:
        if self.connected:
            self.append_chat("[System] Already connected.")
            return

        host = self.host_entry.get().strip()
        port_str = self.port_entry.get().strip()

        try:
            port = int(port_str)
        except ValueError:
            self.append_chat("[Error] Port must be an integer.")
            return

        self.stop_event.clear()

        role = self.role.get()
        if role == "server":
            threading.Thread(
                target=self._run_server, args=(host, port), daemon=True
            ).start()
        else:
            threading.Thread(
                target=self._run_client, args=(host, port), daemon=True
            ).start()

    def _run_server(self, host: str, port: int) -> None:
        try:
            self.set_status("Starting server...", "yellow")
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((host, port))
            srv.listen(1)
            self.server_socket = srv

            LOGGER.info("Server listening on %s:%s", host, port)
            self.append_chat(f"[System] Server listening on {host}:{port}")
            self.set_status("Waiting for client...", "yellow")

            conn, addr = srv.accept()
            self.sock = conn
            self.connected = True
            self.set_status(f"Connected to {addr[0]}:{addr[1]}", "green")
            self.append_chat(f"[System] Client connected from {addr[0]}:{addr[1]}")

            # Perform key exchange handshake
            self._server_handshake()

            # Start receiver thread
            self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self.recv_thread.start()

            # Enable send button
            self.after(0, lambda: self.send_button.configure(state="normal"))

        except Exception as exc:
            LOGGER.exception("Server failed")
            self.append_chat(f"[Error] Server failed: {exc}")
            self.set_status("Server error", "red")
            self._cleanup_sockets()

    def _run_client(self, host: str, port: int) -> None:
        try:
            self.set_status("Connecting...", "yellow")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            self.sock = sock
            self.connected = True
            self.set_status(f"Connected to {host}:{port}", "green")
            self.append_chat(f"[System] Connected to server {host}:{port}")

            # Perform key exchange handshake
            self._client_handshake()

            # Start receiver thread
            self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self.recv_thread.start()

            # Enable send button
            self.after(0, lambda: self.send_button.configure(state="normal"))

        except Exception as exc:
            LOGGER.exception("Client failed to connect")
            self.append_chat(f"[Error] Client failed to connect: {exc}")
            self.set_status("Connection failed", "red")
            self._cleanup_sockets()

    # ----------------- HANDSHAKE (KEY EXCHANGE) -----------------

    def _server_handshake(self) -> None:
        """
        Server side of the RSA + AES key exchange handshake.

        Steps:
          1. Receive client's RSA public key.
          2. Send server's RSA public key.
          3. Generate a random AES key.
          4. Encrypt AES key with client's RSA public key.
          5. Send encrypted AES key to client.

        After this, both server and client share the same AES key
        used for encrypting/decrypting all chat messages.
        """
        try:
            # Step 1: receive client's public key
            if self.sock is None:
                raise RuntimeError("Socket is not available for handshake.")

            msg = recv_message(self.sock)
            if msg.get("type") != "public_key":
                raise ValueError("Expected public_key from client")
            client_pub_pem_b64 = msg["key"]
            client_pub_pem = bytes.fromhex(client_pub_pem_b64)
            self.peer_public_key = deserialize_public_key(client_pub_pem)
            self.append_chat("[System] Received client's RSA public key.")

            # Step 2: send server's public key
            server_pub_pem = serialize_public_key(self.rsa_public_key)
            send_message(
                self.sock,
                {
                    "type": "public_key",
                    "key": server_pub_pem.hex(),
                },
            )
            self.append_chat("[System] Sent server RSA public key to client.")

            # Step 3: generate a random AES key
            self.aes_key = generate_aes_key()

            # Step 4: encrypt AES key with client's RSA public key
            enc_key = rsa_encrypt(self.peer_public_key, self.aes_key)

            # Step 5: send encrypted AES key
            send_message(
                self.sock,
                {
                    "type": "aes_key",
                    "encrypted_key": enc_key.hex(),
                    "from": "server",
                },
            )
            self.append_chat("[System] Encrypted AES key sent to client.")
            self.append_chat("[System] Secure AES channel established.")

        except Exception as exc:
            LOGGER.exception("Handshake (server) failed")
            self.append_chat(f"[Error] Handshake (server) failed: {exc}")
            self.set_status("Handshake failed", "red")
            self._cleanup_sockets()
            raise

    def _client_handshake(self) -> None:
        """
        Client side of the RSA + AES key exchange handshake.

        Steps:
          1. Send client's RSA public key to server.
          2. Receive server's RSA public key.
          3. Receive AES key encrypted with client's RSA public key.
          4. Decrypt AES key with client's RSA private key.

        After this, both server and client share the same AES key
        used for encrypting/decrypting all chat messages.
        """
        try:
            # Step 1: send client's public key
            client_pub_pem = serialize_public_key(self.rsa_public_key)
            send_message(
                self.sock,
                {
                    "type": "public_key",
                    "key": client_pub_pem.hex(),
                },
            )
            self.append_chat("[System] Sent client RSA public key to server.")

            # Step 2: receive server's public key
            if self.sock is None:
                raise RuntimeError("Socket is not available for handshake.")

            msg = recv_message(self.sock)
            if msg.get("type") != "public_key":
                raise ValueError("Expected public_key from server")
            server_pub_pem_b64 = msg["key"]
            server_pub_pem = bytes.fromhex(server_pub_pem_b64)
            self.peer_public_key = deserialize_public_key(server_pub_pem)
            self.append_chat("[System] Received server's RSA public key.")

            # Step 3: receive AES key from server
            msg = recv_message(self.sock)
            if msg.get("type") != "aes_key":
                raise ValueError("Expected aes_key from server")
            enc_key_hex = msg["encrypted_key"]
            enc_key = bytes.fromhex(enc_key_hex)

            # Step 4: decrypt AES key with client's private key
            self.aes_key = rsa_decrypt(self.rsa_private_key, enc_key)

            self.append_chat("[System] Decrypted AES key from server.")
            self.append_chat("[System] Secure AES channel established.")

        except Exception as exc:
            LOGGER.exception("Handshake (client) failed")
            self.append_chat(f"[Error] Handshake (client) failed: {exc}")
            self.set_status("Handshake failed", "red")
            self._cleanup_sockets()
            raise

    # ----------------- RECEIVING LOOP -----------------

    def _recv_loop(self) -> None:
        """
        Background thread: receive messages from the peer asynchronously.

        - Handles chat messages encrypted with AES-GCM.
        - Updates GUI via append_chat (using Tkinter's thread-safe 'after').
        """
        try:
            while not self.stop_event.is_set():
                if self.sock is None:
                    self.append_chat("[System] Socket closed.")
                    break

                try:
                    msg = recv_message(self.sock)
                except ConnectionError:
                    self.append_chat("[System] Connection closed by peer.")
                    break
                except Exception as exc:
                    LOGGER.exception("Receiving message failed")
                    self.append_chat(f"[Error] Receiving message failed: {exc}")
                    break

                msg_type = msg.get("type")

                if msg_type == "chat":
                    # Encrypted chat message
                    try:
                        if self.aes_key is None:
                            raise RuntimeError("AES key not available.")

                        nonce = bytes.fromhex(msg["nonce"])
                        ciphertext = bytes.fromhex(msg["ciphertext"])
                        ts = msg.get("timestamp", "")
                        decrypted = aes_decrypt(self.aes_key, nonce, ciphertext)
                        text = decrypted.decode("utf-8", errors="replace")
                        display_ts = f"[{ts}] " if ts else ""
                        self.append_chat(f"{display_ts}Peer: {text}")
                    except Exception as exc:
                        LOGGER.exception("Failed to decrypt message")
                        self.append_chat(f"[Error] Failed to decrypt message: {exc}")

                elif msg_type == "system":
                    # Plain system notification from peer (if any)
                    content = msg.get("content", "")
                    self.append_chat(f"[Peer-System] {content}")

                else:
                    # Unknown message type
                    self.append_chat(f"[Warning] Unknown message type received: {msg_type}")
        finally:
            self._cleanup_sockets()

    # ----------------- SENDING MESSAGES -----------------

    def on_send_click(self) -> None:
        if not self.connected or self.aes_key is None:
            self.append_chat("[System] Cannot send: not connected or secure channel not ready.")
            return

        text = self.message_entry.get().strip()
        if not text:
            return

        self.message_entry.delete(0, "end")

        try:
            if self.sock is None:
                raise RuntimeError("Socket is not connected.")
            if self.aes_key is None:
                raise RuntimeError("AES key is not available.")

            # Timestamp (optional)
            ts = datetime.datetime.now().strftime("%H:%M:%S")

            # Encrypt message with AES-GCM
            plaintext_bytes = text.encode("utf-8")
            nonce, ciphertext = aes_encrypt(self.aes_key, plaintext_bytes)

            # Send encrypted chat packet
            send_message(
                self.sock,
                {
                    "type": "chat",
                    "nonce": nonce.hex(),
                    "ciphertext": ciphertext.hex(),
                    "timestamp": ts,
                },
            )

            # Display locally
            self.append_chat(f"[{ts}] You: {text}")

        except Exception as exc:
            LOGGER.exception("Failed to send message")
            self.append_chat(f"[Error] Failed to send message: {exc}")

    # ----------------- CLEANUP -----------------

    def _cleanup_sockets(self) -> None:
        """
        Close sockets and update UI state.
        """
        self.stop_event.set()
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
            self.server_socket = None

        if self.connected:
            self.connected = False
            self.set_status("Disconnected", "red")
            self.after(0, lambda: self.send_button.configure(state="disabled"))

    def on_close(self) -> None:
        """
        Called when the window is closed. Ensure we shut down cleanly.
        """
        self.stop_event.set()
        self._cleanup_sockets()
        self.destroy()
        # Ensure the process exits (important if threads are still alive)
        sys.exit(0)


def main() -> None:
    """Entry point for running the application."""
    app = SecureChatApp()
    app.mainloop()


if __name__ == "__main__":
    main()
