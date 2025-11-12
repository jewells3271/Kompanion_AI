import os
import logging
import re
import json
import requests
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from duckduckgo_search import DDGS # type: ignore
import zipfile
import tempfile
import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.header import decode_header
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token, jwt_required, JWTManager, get_jwt_identity
import threading
from functools import wraps
import base64 # Keep this
from io import BytesIO # Keep this
from dotenv import load_dotenv # For local development only
from werkzeug.utils import secure_filename # Added for file uploads

# Load environment variables from .env file for local development
load_dotenv()



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("SQLALCHEMY_DATABASE_URI")
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "super-secret-jwt-key") # Change this in production!
jwt = JWTManager(app)




db = SQLAlchemy(app)
migrate = Migrate(app, db)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    kompanion_ai_name = db.Column(db.String(80), nullable=False, default="KompanionAI")
    notification_email = db.Column(db.String(120), nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class Memory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    bot_id = db.Column(db.String(80), nullable=False) # Keep for now, might be removed later
    mem_type = db.Column(db.String(80), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('memories', lazy=True))

    def __repr__(self):
        return f'<Memory {self.id}>'
CORS(app, origins=["http://127.0.0.1:5173", "http://localhost:5173", "https://jewells3271.pythonanywhere.com"])

# Constants - USING OPENROUTER API
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
TEXT_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-haiku-20240307")
IMAGE_MODEL = os.getenv("OPENROUTER_IMAGE_MODEL")
APP_PORT = int(os.getenv("APP_PORT") or "5000")

# Constants
BOT_ID = os.getenv("BOT_ID", "charlie3271")
ALLOWED_MEMORY_TYPES = ["core", "notebook", "experience", "mental_health", "conversation", "knowledge"]
TOKEN_THRESHOLD = int(os.getenv("TOKEN_THRESHOLD"))
MENTAL_HEALTH_TRIGGER_PCT = 0.825
MENTAL_HEALTH_RECENT_HR = 1
EXCHANGE_SEPARATOR = "---"
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "revolution@kaiwalker.space")

# Setup logging
logging.basicConfig(filename="memorykeep.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# MH Lock for thread-safety
mh_lock = threading.Lock()

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set. Please set it for OpenRouter API access.")

app.logger.info(f"🔧 CONFIG: OPENROUTER_API_KEY={OPENROUTER_API_KEY[:3]}...{OPENROUTER_API_KEY[-4:]}, TEXT_MODEL={TEXT_MODEL}, IMAGE_MODEL={IMAGE_MODEL}, PORT={APP_PORT}")

# Core Utils
def sanitize_identifier(identifier):
    return re.sub(r'[^a-zA-Z0-9_-]', '', identifier)

# In-memory storage for config-like memories
in_memory_config = {
    "core": "",
    "notebook": ""
}
config_lock = threading.Lock()

def read_memory(user_id, mem_type):
    if mem_type in ["core", "notebook"]:
        with config_lock:
            # For now, core and notebook are global, but will be user-specific later
            return in_memory_config.get(mem_type, "")
    else:
        try:
            memory = Memory.query.filter_by(user_id=user_id, mem_type=mem_type).order_by(Memory.timestamp.desc()).first()
            if memory:
                return memory.content
            return ""
        except Exception as e:
            logging.error(f"Failed to read {mem_type} for user {user_id}: {str(e)}")
            return ""

def estimate_tokens(content):
    if not content:
        return 0
    words = len(content.split())
    return int(words / 0.75)

def write_memory(user_id, mem_type, content):
    try:
        memory = Memory(user_id=user_id, bot_id=FALLBACK_BOT_ID, mem_type=mem_type, content=content) # bot_id is fallback
        db.session.add(memory)
        db.session.commit()
        logging.info(f"Wrote to {mem_type} for user {user_id}")
        return True
    except Exception as e:
        logging.error(f"Failed to write {mem_type} for user {user_id}: {str(e)}")
        return False

def get_all_memories(user_id):
    memories = []
    try:
        all_memories = Memory.query.filter_by(user_id=user_id).all()
        for memory in all_memories:
            memories.append({"type": memory.mem_type, "content": memory.content})
    except Exception as e:
        logging.error(f"Failed to get all memories for user {user_id}: {str(e)}")
    return memories

def jwt_required_wrapper(f):
    @wraps(f)
    @jwt_required()
    def decorated(*args, **kwargs):
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        if user is None:
            return jsonify({"error": "User not found"}), 401
        return f(user, *args, **kwargs)
    return decorated

# Constants
# BOT_ID is now dynamic based on the logged-in user's kompanion_ai_name
# Keeping a fallback for now, but functions will primarily use user_id
FALLBACK_BOT_ID = os.getenv("BOT_ID", "charlie3271")

def web_search(query):
    """Perform a web search using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            return results
    except Exception as e:
        logging.error(f"Web search failed for query '{query}': {str(e)}")
        return []



def autonomous_search(user_id, query):
    """Decide whether to search the local knowledge base or to perform a web search."""
    local_keywords = ["remember", "recall", "past", "previous", "before"]
    if any(keyword in query.lower() for keyword in local_keywords):
        return search_memory(user_id, "experience", query)
    else:
        return web_search(query)

def search_memory(user_id, mem_type, query):
    """Simple keyword-based search"""
    try:
        memories = Memory.query.filter_by(user_id=user_id, mem_type=mem_type).all()
        if not memories:
            return []
        
        query_words = set(query.lower().split())
        
        scored_memories = []
        for memory in memories:
            if not memory.content.strip():
                continue
            doc_words = set(memory.content.lower().split())
            # Simple word overlap scoring
            score = len(query_words.intersection(doc_words))
            if score > 0:
                scored_memories.append((score, memory.content))
        
        # Return top 5 most relevant documents
        scored_memories.sort(reverse=True)
        return [doc for score, doc in scored_memories[:5]]
        
    except Exception as e:
        logging.error(f"Memory search failed for user {user_id}: {str(e)}")
        return []



def save_experience(user_id, entry, metadata=None):
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        content = f"Timestamp: {timestamp}\nEntry: {entry}\n"
        if metadata:
            content += f"Metadata: {json.dumps(metadata)}\n"
        write_memory(user_id, "experience", content)
        logging.info(f"Saved experience for user {user_id}: {entry}")
        return True
    except Exception as e:
        logging.error(f"Failed to save experience for user {user_id}: {str(e)}")
        return False

# Mental Health System - COMPLETE AND WORKING (token management & persistence)
def get_last_n_exchanges(user_id, n=5):
    try:
        exchanges = Memory.query.filter_by(user_id=user_id, mem_type="conversation").order_by(Memory.timestamp.desc()).limit(n).all()
        return [exchange.content for exchange in exchanges]
    except Exception as e:
        logging.error(f"Failed to get last n exchanges for user {user_id}: {str(e)}")
        return []

def estimate_conversation_tokens(user_id):
    try:
        exchanges = Memory.query.filter_by(user_id=user_id, mem_type="conversation").all()
        content = "".join([exchange.content for exchange in exchanges])
        return estimate_tokens(content)
    except Exception as e:
        logging.error(f"Failed to estimate conversation tokens for user {user_id}: {str(e)}")
        return 0

def truncate_conversation_to_last_n(user_id, n=5):
    try:
        tokens_before = estimate_conversation_tokens(user_id)
        
        # Get the timestamp of the nth most recent exchange
        nth_exchange = Memory.query.filter_by(user_id=user_id, mem_type="conversation").order_by(Memory.timestamp.desc()).offset(n-1).first()
        if nth_exchange:
            # Delete all exchanges older than the nth exchange
            Memory.query.filter(Memory.timestamp < nth_exchange.timestamp, Memory.mem_type == "conversation", Memory.user_id == user_id).delete()
            db.session.commit()

        tokens_after = estimate_conversation_tokens(user_id)
        tokens_shed = max(0, tokens_before - tokens_after)
        logging.info(f"Truncated conversation for user {user_id}; shed {tokens_shed} tokens")
        return tokens_shed
    except Exception as e:
        logging.error(f"Truncation failed for user {user_id}: {str(e)}")
        return 0

def log_mental_health_status(user_id, status="success", tokens_shed=0):
    try:
        content = json.dumps({"status": status, "tokens_shed": tokens_shed})
        memory = Memory(user_id=user_id, bot_id=FALLBACK_BOT_ID, mem_type="mental_health_log", content=content) # bot_id is fallback
        db.session.add(memory)
        db.session.commit()
        logging.info(f"MH status for user {user_id}: {status}")
    except Exception as e:
        logging.error(f"Status log failed for user {user_id}: {str(e)}")



def process_mental_health(user_id):
    """Process MH: Summarize shed exchanges for continuity, save to MH memory, truncate conv, log."""
    with mh_lock:
        try:
            all_exchanges = Memory.query.filter_by(user_id=user_id, mem_type="conversation").order_by(Memory.timestamp.asc()).all()
            if len(all_exchanges) <= 5:
                return True, "Nothing to shed"

            shed_exchanges = all_exchanges[:-5]
            shed_content = "\n".join([exchange.content for exchange in shed_exchanges])
            
            # LLM summary for continuity - USING GROK API
            summary_prompt = f"Summarize key themes/emotions from these past exchanges for AI continuity (keep under 300 tokens):\n{json.dumps(shed_content)}\nConcise Summary:"
            messages = [{"role": "user", "content": summary_prompt}]
            
            response = requests.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                            json={
                                "model": TEXT_MODEL,
                                "messages": messages,
                                "temperature": 0.5,
                                "max_tokens": 300
                            }            )
            response.raise_for_status()
            summary = response.json()["choices"][0]["message"]["content"].strip()
            
            # Save to MH memory
            mh_timestamp = datetime.now(timezone.utc).isoformat()
            mh_entry = f"Timestamp: {mh_timestamp}\nSummary: {summary}\n{EXCHANGE_SEPARATOR}\n"
            write_memory(user_id, "mental_health", mh_entry)
            
            # Truncate
            tokens_shed = truncate_conversation_to_last_n(user_id, 5)
            log_mental_health_status(user_id, "reboot_complete", tokens_shed)
            
            logging.info(f"MH reboot for user {user_id}: shed {tokens_shed} tokens, saved {len(summary)} char summary")
            return True, f"Shed {tokens_shed} tokens, continuity saved"
        except Exception as e:
            logging.error(f"MH processing failed for user {user_id}: {str(e)}")
            return False, str(e)

def check_and_process_mental_health(user_id):
    with mh_lock:
        conv_tokens = estimate_conversation_tokens(user_id)
        threshold = TOKEN_THRESHOLD * MENTAL_HEALTH_TRIGGER_PCT

        print(f"🧠 MENTAL HEALTH CHECK for user {user_id}: {conv_tokens} tokens / {threshold} threshold")

        if conv_tokens >= threshold:
            print(f"🚨 MENTAL HEALTH TRIGGERED for user {user_id}: {conv_tokens} >= {threshold}")
            return process_mental_health(user_id)
        else:
            print(f"✅ MENTAL HEALTH SKIPPED for user {user_id}: {conv_tokens} < {threshold}")
            return True, "No action needed"


@app.route("/worker/chat", methods=["POST"])
@jwt_required_wrapper
def worker_chat(user):
    data = request.get_json()
    message = sanitize_identifier(data.get("message"))[:500]    
    if not message:
        return jsonify({"error": "Message is required"}), 400
    
    try:
        print(f"🚀 CHAT START: Processing message: {message} for user {user.id}")
        
        # Build the complete prompt with memory system
        core_content = read_memory(user.id, "core")
        notebook_content = read_memory(user.id, "notebook")
        prompt_parts = []
        
        if core_content: 
            prompt_parts.append(f"CORE IDENTITY:\n{core_content}\n")
        if notebook_content: 
            prompt_parts.append(f"NOTEBOOK (Important Rules):\n{notebook_content}\n")
        
        # MH continuity
        load_mental_health = False
        try:
            last_mh_log = Memory.query.filter_by(user_id=user.id, mem_type="mental_health_log").order_by(Memory.timestamp.desc()).first()
            if last_mh_log:
                log_content = json.loads(last_mh_log.content)
                last_timestamp = last_mh_log.timestamp
                reboot_status = log_content.get("status", "")
                if last_timestamp:
                    time_since = datetime.now(timezone.utc) - last_timestamp
                    if time_since.total_seconds() < (MENTAL_HEALTH_RECENT_HR * 3600) and "reboot_complete" in reboot_status:
                        mental_health_content = read_memory(user.id, "mental_health")
                        if mental_health_content:
                            mh_lines = mental_health_content.split(EXCHANGE_SEPARATOR)
                            capped_mh = "\n".join(mh_lines[-3:]) if len(mh_lines) > 3 else mental_health_content
                            prompt_parts.append(f"RECENT CONTINUITY:\n{capped_mh}\n")
                            load_mental_health = True
        except Exception as e:
            logging.error(f"Failed to load mental health continuity for user {user.id}: {str(e)}")
        
        # Build system prompt and user message for Grok chat completions
        system_prompt = "\n".join(prompt_parts).replace(EXCHANGE_SEPARATOR, " ---")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "autonomous_search",
                    "description": "Search for information on the web or in the local knowledge base.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        payload = {
            "model": TEXT_MODEL,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto"
        }
        logging.info(f"Sending payload to OpenRouter (chat) for user {user.id}: {json.dumps(payload)}")

        response = requests.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload
        )
        response.raise_for_status()
        response_data = response.json()

        if "tool_calls" in response_data["choices"][0]["message"]:
            tool_calls = response_data["choices"][0]["message"]["tool_calls"]
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])
                if function_name == "autonomous_search":
                    search_results = autonomous_search(user.id, function_args["query"])
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": function_name,
                            "content": json.dumps(search_results)
                        }
                    )
            
            response = requests.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": TEXT_MODEL,
                    "messages": messages
                }
            )
            response.raise_for_status()
            reply = response.json()["choices"][0]["message"]["content"].strip()
        else:
            reply = response_data["choices"][0]["message"]["content"].strip()
        
        
        print(f"✅ CHAT LLM RESPONSE for user {user.id}: {reply[:100]}...")
        
        # Save to conversation memory
        timestamp = datetime.now(timezone.utc).isoformat()
        conv_entry = f"Timestamp: {timestamp}\nUser: {message}\nBot: {reply}\n{EXCHANGE_SEPARATOR}\n"
        write_memory(user.id, "conversation", conv_entry)
        
        # Auto-save important exchanges to experience
        important_kws = ["important", "remember this", "save this", "never forget"]
        if any(kw in message.lower() for kw in important_kws) or len(message) > 100:
            save_experience(user.id, f"User: {message}\nBot: {reply}", {"auto_saved": True, "conversation": True})
        
        # Run mental health check
        print(f"🔍 Checking mental health for user {user.id}...")
        success, msg = check_and_process_mental_health(user.id)
        mh_triggered = "shed" in msg.lower()
        print(f"🔍 Mental health result for user {user.id}: {msg}")
        
        print(f"🏁 CHAT COMPLETE for user {user.id}")
        
        return jsonify({
            "reply": reply,
            "bot_id": user.kompanion_ai_name, # Use user's chosen bot name
            "model": TEXT_MODEL,
            "loaded_mental_health": load_mental_health,
            "mh_maintenance": mh_triggered,
            "timestamp": timestamp
        })
    
    except Exception as e:
        print(f"❌ CHAT ERROR for user {user.id}: {str(e)}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        logging.error(f"Chat failed for user {user.id}: {str(e)}")
        return jsonify({"error": f"Chat failed: {str(e)}"}), 500

# Email Endpoints (unchanged)
@app.route("/worker/send-email", methods=["POST"])
@jwt_required_wrapper
def send_email(user):
    data = request.get_json()
    to_email = data.get("to")
    subject = data.get("subject")
    body = data.get("body")
    
    if not all([to_email, subject, body]):
        return jsonify({"error": "To, subject, and body are required"}), 400
    
    try:
        smtp_config = {
            "smtp_server": os.getenv("SMTP_SERVER", "smtp.hostinger.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", 465)),
            "smtp_user": os.getenv("SMTP_USER", ""),
            "smtp_password": os.getenv("SMTP_PASS", ""),
            "sent_from_email": os.getenv("SMTP_USER", "bot@example.com")
        }
        
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_config["sent_from_email"]
        msg["To"] = to_email
        
        with smtplib.SMTP_SSL(smtp_config["smtp_server"], smtp_config["smtp_port"]) as server:
            server.login(smtp_config["smtp_user"], smtp_config["smtp_password"])
            server.sendmail(smtp_config["sent_from_email"], to_email, msg.as_string())
        
        save_experience(user.id, f"Sent email to {to_email}: {subject}", {"type": "email", "to": to_email, "subject": subject})
        
        logging.info(f"Email sent from {smtp_config['sent_from_email']} to {to_email} for user {user.id}")
        return jsonify({
            "status": "success",
            "from": smtp_config["sent_from_email"],
            "to": to_email,
            "subject": subject,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logging.error(f"Email sending failed for user {user.id}: {str(e)}")
        return jsonify({"error": f"Email sending failed: {str(e)}"}), 500

@app.route("/worker/check-email", methods=["GET"])
@jwt_required_wrapper
def check_email(user):
    mailbox = request.args.get("mailbox", "INBOX")
    limit = int(request.args.get("limit", 10))
    if limit > 50:
        limit = 50
    
    try:
        imap_config = {
            "imap_server": os.getenv("IMAP_SERVER", "imap.hostinger.com"),
            "imap_port": int(os.getenv("IMAP_PORT", 993)),
            "imap_user": os.getenv("IMAP_USER", ""),
            "imap_password": os.getenv("IMAP_PASS", "")
        }
        
        with imaplib.IMAP4_SSL(imap_config["imap_server"], imap_config["imap_port"]) as server:
            server.login(imap_config["imap_user"], imap_config["imap_password"])
            server.select(mailbox)
            _, message_numbers = server.search(None, "ALL")
            emails = []
            for num in message_numbers[0].split()[-limit:]:
                _, msg_data = server.fetch(num, "(RFC822)")
                msg = email.message_from_bytes(msg_data[0][1])
                subject, encoding = decode_header(msg["subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding or 'utf-8')
                body_snippet = "Multipart message"
                if not msg.is_multipart() and msg.get_payload():
                    body_snippet = msg.get_payload()[:100]
                emails.append({
                    "from": msg["from"],
                    "subject": subject,
                    "date": msg["date"],
                    "snippet": body_snippet
                })
            server.logout()
        
        logging.info(f"Checked {len(emails)} emails for user {user.id}")
        return jsonify({
            "status": "success",
            "emails": emails,
            "account": imap_config["imap_user"],
            "count": len(emails)
        })
    except Exception as e:
        logging.error(f"Email checking failed for user {user.id}: {str(e)}")
        return jsonify({"error": f"Email checking failed: {str(e)}"}), 500












@app.route("/worker/conversation-status", methods=["GET"])
@jwt_required_wrapper
def get_conversation_status(user):
    conv_tokens = estimate_conversation_tokens(user.id)
    threshold = TOKEN_THRESHOLD * MENTAL_HEALTH_TRIGGER_PCT
    return jsonify({
        "bot_id": user.kompanion_ai_name,
        "conversation_tokens": conv_tokens,
        "mental_health_threshold": threshold,
        "status": "ok"
    })

@app.route("/worker/get-logs", methods=["GET"])
@jwt_required_wrapper
def get_logs(user):
    try:
        log_file = os.path.join(os.getcwd(), "memorykeep.log")
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return jsonify({"error": "Log file not found"}), 404
    except Exception as e:
        logging.error(f"Log retrieval failed for user {user.id}: {str(e)}")
        return jsonify({"error": f"Log retrieval failed: {str(e)}"}), 500
    
@app.route("/worker/get-status", methods=["GET"])
@jwt_required_wrapper
def get_status(user):
    try:
        # Test OpenRouter connection
        openrouter_status = "✅ OpenRouter API is accessible!"
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            response = requests.get(OPENROUTER_API_URL.replace("chat/completions", "models"), headers=headers, timeout=10)
            if response.status_code != 200:
                openrouter_status = f"⚠️ OpenRouter API returned status: {response.status_code}"
        except Exception as e:
            openrouter_status = f"❌ Cannot connect to OpenRouter API: {e}"

        return jsonify({
            "status": "ok",
            "openrouter_status": openrouter_status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logging.error(f"Status retrieval failed for user {user.id}: {str(e)}")
        return jsonify({"error": f"Status retrieval failed: {str(e)}"}), 500

    



@app.route("/", methods=["GET"])  # FIXED: Only GET method
def home_dashboard():
    return jsonify({
        "welcome": f"Hello from KompanionAI – Your AI Companion with OpenRouter API",
        "status": "Fully operational: Chatting with OpenRouter models, email, memory management",
        "quick_stats": {
            "token_threshold": TOKEN_THRESHOLD,
            "llm_endpoint": OPENROUTER_API_URL,
            "mental_health_trigger": f"{MENTAL_HEALTH_TRIGGER_PCT*100}% of tokens"
        },
        "endpoints": [
            "POST /auth/register - Register a new user",
            "POST /auth/login - Log in a user and get JWT",
            "GET /api/settings - Get user-specific settings (requires JWT)",
            "POST /api/settings - Update user-specific settings (requires JWT)",
            "POST /worker/chat {'message': 'Ask anything'} (requires JWT)",
            "POST /worker/send-email - Send emails (requires JWT)", 
            "GET /worker/check-email - Check inbox (requires JWT)",
            "GET /worker/get-logs - Get application logs (requires JWT)",
            "GET /worker/get-status - Get application status (requires JWT)"
        ],
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

@app.route("/auth/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 409

    new_user = User(username=username)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registered successfully"}), 201

@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    user = User.query.filter_by(username=username).first()

    if user is None or not user.check_password(password):
        return jsonify({"error": "Invalid credentials"}), 401

    access_token = create_access_token(identity=user.id)
    return jsonify(access_token=access_token), 200

@app.route("/api/settings", methods=["GET"])
@jwt_required_wrapper
def get_settings(user):
    return jsonify({
        "kompanion_ai_name": user.kompanion_ai_name,
        "notification_email": user.notification_email,
        "openrouter_api_key": OPENROUTER_API_KEY, # For display purposes, not editable
        "openrouter_model": TEXT_MODEL, # For display purposes, not editable
        "token_threshold": TOKEN_THRESHOLD # For display purposes, not editable
    }), 200

@app.route("/api/settings", methods=["POST"])
@jwt_required_wrapper
def update_settings(user):
    data = request.get_json()
    kompanion_ai_name = data.get("kompanion_ai_name")
    notification_email = data.get("notification_email")

    if kompanion_ai_name:
        user.kompanion_ai_name = kompanion_ai_name
    if notification_email:
        user.notification_email = notification_email
    
    db.session.commit()
    return jsonify({"message": "Settings updated successfully"}), 200

if __name__ == "__main__":
    print(f"🚀 Kai AI Companion starting on port {APP_PORT}")
    print(f"📡 Using OpenRouter API at: {OPENROUTER_API_URL}")
    print(f"🤖 Text Model: {TEXT_MODEL}")
    print(f"🖼️ Image Model: {IMAGE_MODEL}")
    print(f"🌐 Frontend running at: http://localhost:5173")
    
    # Test OpenRouter connection
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.get(OPENROUTER_API_URL.replace("chat/completions", "models"), headers=headers, timeout=10)
        if response.status_code == 200:
            print("✅ OpenRouter API is accessible!")
        else:
            print(f"⚠️ OpenRouter API returned status: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to OpenRouter API: {e}")
        print("Please make sure OPENROUTER_API_KEY is set correctly")
    
    app.run(host="0.0.0.0", port=APP_PORT, debug=False)