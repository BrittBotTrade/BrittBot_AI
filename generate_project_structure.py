import os

# Helper to write file content
def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

# --------- FILE CONTENTS FROM YOUR index.html ---------

env_content = """# Alpaca Trading API Credentials
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
# Use https://api.alpaca.markets for live trading

# WebSocket Configuration  
WS_SERVER_URL=ws://localhost:8765

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=brittbot.log

# Risk Management
MAX_POSITION_SIZE=0.05
DAILY_LOSS_LIMIT=0.02
"""

gitignore_content = """# Environment and secrets
.env
*.secret
config.py

# Python cache
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python

# Virtual environment
venv/
env/
ENV/

# Logs
*.log
logs/

# Database
*.sqlite3
*.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Trading specific
brittbot_signal_log.json
shared_trades.json
"""

readme_content = "# BrittBot Project\nSee index.html for full documentation and critical notes."

runpy_content = """#!/usr/bin/env python3
\"\"\"
BrittBot Main Runner
Starts AI engine and WebSocket services
\"\"\"

import os
import sys
import threading
import time
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.getenv('LOG_FILE', 'brittbot.log'))
    ]
)

logger = logging.getLogger(__name__)

def validate_environment():
    \"\"\"Validate required environment variables\"\"\"
    required_vars = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please check your .env file")
        sys.exit(1)

def start_ai_service():
    \"\"\"Start the AI trading service\"\"\"
    try:
        from BrittBot_AI.main import start_ai_service
        logger.info("Starting AI trading service...")
        start_ai_service()
    except Exception as e:
        logger.error(f"Failed to start AI service: {e}")
        sys.exit(1)

def start_client_service():
    \"\"\"Start the WebSocket client service\"\"\"
    try:
        from BrittBot_Setup.client import start_client
        logger.info("Starting WebSocket client...")
        start_client()
    except Exception as e:
        logger.error(f"Failed to start client service: {e}")

def main():
    \"\"\"Main entry point\"\"\"
    logger.info("=" * 50)
    logger.info("BrittBot Trading System Starting...")
    logger.info("=" * 50)
    
    # Validate environment
    validate_environment()
    
    # Start services in separate threads
    ai_thread = threading.Thread(target=start_ai_service, daemon=True)
    client_thread = threading.Thread(target=start_client_service, daemon=True)
    
    ai_thread.start()
    client_thread.start()
    
    logger.info("All services started. System is operational.")
    logger.info("Press Ctrl+C to stop the system")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down BrittBot system...")
        sys.exit(0)

if __name__ == "__main__":
    main()
"""

mainpy_content = """\"\"\"
BrittBot AI Engine
Core AI trading logic with WebSocket server
\"\"\"

import os
import asyncio
import websockets
import json
import threading
import time
import logging
from flask import Flask, jsonify
from .ai_integration import SmartTradingEngine
from .monitoring_utils import log_trade, performance_snapshot

logger = logging.getLogger(__name__)
app = Flask(__name__)

# Global AI engine instance
ai_engine = None

class AITradingService:
    def __init__(self):
        self.ai_engine = SmartTradingEngine()
        self.is_running = False
        self.websocket_clients = set()
    
    async def websocket_server(self, websocket, path):
        \"\"\"Handle WebSocket connections\"\"\"
        self.websocket_clients.add(websocket)
        logger.info(f"New WebSocket client connected. Total clients: {len(self.websocket_clients)}")
        
        try:
            await websocket.wait_closed()
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.websocket_clients.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total clients: {len(self.websocket_clients)}")
    
    async def broadcast_signal(self, signal):
        \"\"\"Broadcast trading signal to all connected clients\"\"\"
        if not self.websocket_clients:
            return
        
        message = json.dumps(signal)
        disconnected = set()
        
        for websocket in self.websocket_clients:
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected
    
    def generate_trading_signal(self, symbol):
        \"\"\"Generate AI trading signal for symbol\"\"\"
        try:
            # This would be replaced with actual AI logic
            import pandas as pd
            import random
            
            # Simulate price data
            price_data = pd.DataFrame({
                'close': [100 + random.uniform(-5, 5) for _ in range(50)]
            })
            
            action, details = self.ai_engine.smart_trade_decision(
                symbol, price_data, []
            )
            
            signal = {
                'timestamp': time.time(),
                'symbol': symbol,
                'action': action,
                'details': details,
                'source': 'AI_Engine'
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    async def ai_trading_loop(self):
        \"\"\"Main AI trading loop\"\"\"
        symbols = ['AAPL', 'TSLA', 'AMZN', 'MSFT', 'GOOGL', 'META', 'NVDA', 'NFLX']
        
        while self.is_running:
            try:
                for symbol in symbols:
                    signal = self.generate_trading_signal(symbol)
                    if signal:
                        await self.broadcast_signal(signal)
                        log_trade(signal)
                        logger.info(f"Generated signal: {symbol} -> {signal['action']}")
                
                await asyncio.sleep(10)  # Wait 10 seconds between cycles
                
            except Exception as e:
                logger.error(f"Error in AI trading loop: {e}")
                await asyncio.sleep(5)
    
    async def start_services(self):
        \"\"\"Start all async services\"\"\"
        self.is_running = True
        
        # Start WebSocket server
        websocket_server = websockets.serve(
            self.websocket_server, 
            "0.0.0.0", 
            8765
        )
        
        logger.info("WebSocket server started on ws://0.0.0.0:8765")
        
        # Start both services concurrently
        await asyncio.gather(
            websocket_server,
            self.ai_trading_loop()
        )

# Flask routes
@app.route('/status')
def status():
    \"\"\"Get AI engine status\"\"\"
    global ai_engine
    if ai_engine:
        return jsonify(ai_engine.ai_engine.get_system_status())
    return jsonify({'error': 'AI engine not initialized'}), 500

@app.route('/health')
def health():
    \"\"\"Health check endpoint\"\"\"
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'clients_connected': len(ai_engine.websocket_clients) if ai_engine else 0
    })

def start_flask_server():
    \"\"\"Start Flask server in separate thread\"\"\"
    app.run(host='0.0.0.0', port=5000, debug=False)

def start_ai_service():
    \"\"\"Start the AI service (called from run.py)\"\"\"
    global ai_engine
    
    ai_engine = AITradingService()
    
    # Start Flask server in separate thread
    flask_thread = threading.Thread(target=start_flask_server, daemon=True)
    flask_thread.start()
    
    # Start async services
    asyncio.run(ai_engine.start_services())
"""

aiintegrationpy_content = """\"\"\"
AI Trading Engine
Core AI logic for trading decisions
\"\"\"

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import json
import os

logger = logging.getLogger(__name__)

class PatternRecognizer:
    \"\"\"Technical pattern recognition\"\"\"
    
    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> Dict:
        \"\"\"Detect technical patterns in price data\"\"\"
        if df.empty or len(df) < 20:
            return {'patterns_detected': []}
        
        patterns = []
        
        try:
            # Simple moving average crossover
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            
            if df['sma_10'].iloc[-1] > df['sma_20'].iloc[-1]:
                patterns.append('bullish_crossover')
            elif df['sma_10'].iloc[-1] < df['sma_20'].iloc[-1]:
                patterns.append('bearish_crossover')
            
            # RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            if rsi.iloc[-1] > 70:
                patterns.append('overbought')
            elif rsi.iloc[-1] < 30:
                patterns.append('oversold')
            
        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")
        
        return {'patterns_detected': patterns}

class SentimentAnalyzer:
    \"\"\"Sentiment analysis for news/headlines\"\"\"
    
    def analyze_sentiment(self, headlines: List[str]) -> Dict:
        \"\"\"Analyze sentiment of headlines\"\"\"
        if not headlines:
            return {'sentiment_score': 0.5, 'confidence': 0.0}
        
        # Simplified sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'positive', 'up', 'gain', 'profit']
        negative_words = ['bad', 'terrible', 'negative', 'down', 'loss', 'decline', 'drop']
        
        total_score = 0
        word_count = 0
        
        for headline in headlines:
            words = headline.lower().split()
            for word in words:
                if word in positive_words:
                    total_score += 1
                    word_count += 1
                elif word in negative_words:
                    total_score -= 1
                    word_count += 1
        
        if word_count == 0:
            return {'sentiment_score': 0.5, 'confidence': 0.0}
        
        # Normalize to 0-1 scale
        sentiment_score = max(0, min(1, (total_score / word_count + 1) / 2))
        confidence = min(1.0, word_count / 10)
        
        return {
            'sentiment_score': sentiment_score,
            'confidence': confidence
        }

class RiskManager:
    \"\"\"Risk management and position sizing\"\"\"
    
    def __init__(self):
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', 0.05))
        self.daily_loss_limit = float(os.getenv('DAILY_LOSS_LIMIT', 0.02))
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_reset = datetime.now().date()
    
    def check_risk_limits(self, position_size: float, current_pnl: float) -> bool:
        \"\"\"Check if trade passes risk limits\"\"\"
        # Reset daily counters if new day
        if datetime.now().date() > self.last_reset:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset = datetime.now().date()
        
        # Check position size limit
        if position_size > self.max_position_size:
            logger.warning(f"Position size {position_size} exceeds limit {self.max_position_size}")
            return False
        
        # Check daily loss limit
        if self.daily_pnl + current_pnl < -self.daily_loss_limit:
            logger.warning(f"Daily loss limit would be exceeded")
            return False
        
        return True
    
    def calculate_position_size(self, confidence: float, account_value: float) -> float:
        \"\"\"Calculate position size based on confidence and risk limits\"\"\"
        base_size = self.max_position_size * confidence
        return min(base_size, self.max_position_size)

class SmartTradingEngine:
    \"\"\"Main AI trading engine\"\"\"
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pattern_recognizer = PatternRecognizer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.risk_manager = RiskManager()
        
        # Load configuration
        self.config = self._load_config()
        
        # RL agent state
        self.rl_agent = {
            'trade_count': 0,
            'win_rate': 0.0,
            'total_trades': 0,
            'winning_trades': 0
        }
        
        self.logger.info("SmartTradingEngine initialized")
    
    def _load_config(self) -> Dict:
        \"\"\"Load AI configuration\"\"\"
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'ai_config.json')
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {
                'sentiment_weight': 0.3,
                'pattern_weight': 0.4,
                'rl_weight': 0.3,
                'min_confidence_threshold': 0.6
            }
    
    def smart_trade_decision(
        self, 
        symbol: str, 
        price_data: pd.DataFrame, 
        headlines: List[str]
    ) -> Tuple[str, Dict]:
        \"\"\"Make AI-powered trading decision\"\"\"
        
        try:
            # Pattern analysis
            pattern_analysis = self.pattern_recognizer.detect_patterns(price_data)
            
            # Sentiment analysis
            sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(headlines)
            
            # Calculate composite score
            pattern_score = self._calculate_pattern_score(pattern_analysis)
            sentiment_score = sentiment_analysis['sentiment_score']
            rl_score = self._get_rl_score(symbol)
            
            # Weighted combination
            composite_score = (
                pattern_score * self.config['pattern_weight'] +
                sentiment_score * self.config['sentiment_weight'] +
                rl_score * self.config['rl_weight']
            )
            
            # Decision logic
            action = 'hold'
            confidence = abs(composite_score - 0.5) * 2  # Convert to 0-1 confidence
            
            if composite_score > 0.6 and confidence >= self.config['min_confidence_threshold']:
                action = 'buy'
            elif composite_score < 0.4 and confidence >= self.config['min_confidence_threshold']:
                action = 'sell'
            
            # Risk check
            position_size = self.risk_manager.calculate_position_size(confidence, 100000)  # Assume $100k account
            
            if not self.risk_manager.check_risk_limits(position_size, 0):
                action = 'hold'
            
            details = {
                'confidence': confidence,
                'composite_score': composite_score,
                'trade_params': {
                    'position_size': position_size,
                    'stop_loss': 0.02,
                    'take_profit': 0.05
                },
                'scores': {
                    'sentiment': sentiment_score,
                    'pattern': pattern_score,
                    'rl': rl_score
                },
                'analysis': {
                    'patterns': pattern_analysis,
                    'sentiment': sentiment_analysis
                }
            }
            
            return action, details
            
        except Exception as e:
            self.logger.error(f"Error in smart_trade_decision: {e}")
            return 'hold', {'error': str(e), 'confidence': 0.0}
    
    def _calculate_pattern_score(self, pattern_analysis: Dict) -> float:
        \"\"\"Convert pattern analysis to score\"\"\"
        patterns = pattern_analysis.get('patterns_detected', [])
        
        if not patterns:
            return 0.5  # Neutral
        
        bullish_patterns = ['bullish_crossover', 'oversold']
        bearish_patterns = ['bearish_crossover', 'overbought']
        
        score = 0.5
        for pattern in patterns:
            if pattern in bullish_patterns:
                score += 0.1
            elif pattern in bearish_patterns:
                score -= 0.1
        
        return max(0, min(1, score))
    
    def _get_rl_score(self, symbol: str) -> float:
        \"\"\"Get reinforcement learning score\"\"\"
        # Simplified RL - would be more sophisticated in real implementation
        if self.rl_agent['total_trades'] == 0:
            return 0.5
        
        return min(1.0, self.rl_agent['win_rate'])
    
    def record_trade_outcome(
        self, 
        symbol: str, 
        entry_price: float, 
        exit_price: float, 
        trade_details: Dict
    ):
        \"\"\"Record trade outcome for RL learning\"\"\"
        try:
            self.rl_agent['total_trades'] += 1
            
            # Determine if trade was winning
            action = trade_details.get('action', 'hold')
            profit = 0
            
            if action == 'buy':
                profit = exit_price - entry_price
            elif action == 'sell':
                profit = entry_price - exit_price
            
            if profit > 0:
                self.rl_agent['winning_trades'] += 1
            
            # Update win rate
            self.rl_agent['win_rate'] = (
                self.rl_agent['winning_trades'] / self.rl_agent['total_trades']
            )
            
            self.logger.info(
                f"Recorded trade outcome for {symbol}: "
                f"Profit={profit:.4f}, Win Rate={self.rl_agent['win_rate']:.3f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error recording trade outcome: {e}")
    
    def get_system_status(self) -> Dict:
        \"\"\"Get current system status\"\"\"
        return {
            'rl_agent': self.rl_agent,
            'risk_limits_ok': True,
            'daily_stats': {
                'trades_today': self.risk_manager.daily_trades,
                'daily_pnl': self.risk_manager.daily_pnl
            },
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
"""

clientpy_content = """\"\"\"
WebSocket Client for BrittBot
Connects to AI engine and processes trading signals
\"\"\"

import asyncio
import websockets
import json
import os
import logging
from datetime import datetime
from .trading_client import TradingClient

logger = logging.getLogger(__name__)

class BrittBotClient:
    def __init__(self):
        self.ws_url = os.getenv('WS_SERVER_URL', 'ws://localhost:8765')
        self.log_file = 'brittbot_signal_log.json'
        self.trading_client = TradingClient()
        self.is_running = False
    
    async def connect_and_listen(self):
        \"\"\"Connect to WebSocket server and listen for signals\"\"\"
        logger.info(f"Connecting to WebSocket server at {self.ws_url}")
        
        while self.is_running:
            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    
                    logger.info("✅ Connected to WebSocket server")
                    
                    async for message in websocket:
                        try:
                            signal = json.loads(message)
                            await self.process_signal(signal)
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON received: {e}")
                        except Exception as e:
                            logger.error(f"Error processing signal: {e}")
            
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed. Reconnecting in 5s...")
                await asyncio.sleep(5)
            
            except websockets.exceptions.InvalidURI:
                logger.error(f"Invalid WebSocket URL: {self.ws_url}")
                break
            
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}. Retrying in 5s...")
                await asyncio.sleep(5)
    
    async def process_signal(self, signal: dict):
        \"\"\"Process incoming trading signal\"\"\"
        try:
            # Validate signal
            if not self._validate_signal(signal):
                return
            
            # Log signal
            self._log_signal(signal)
            
            # Execute trade if conditions are met
            if signal.get('action') in ['buy', 'sell']:
                await self._execute_trade(signal)
            
            logger.info(
                f"📈 Signal processed: {signal['symbol']} -> {signal['action']} "
                f"(confidence: {signal.get('details', {}).get('confidence', 0):.3f})"
            )
            
        except Exception as e:
            logger.error(f"Error in process_signal: {e}")
    
    def _validate_signal(self, signal: dict) -> bool:
        \"\"\"Validate trading signal structure\"\"\"
        required_fields = ['timestamp', 'symbol', 'action', 'source']
        
        for field in required_fields:
            if field not in signal:
                logger.error(f"Missing required field in signal: {field}")
                return False
        
        if signal['action'] not in ['buy', 'sell', 'hold']:
            logger.error(f"Invalid action in signal: {signal['action']}")
            return False
        
        return True
    
    def _log_signal(self, signal: dict):
        \"\"\"Log signal to file\"\"\"
        try:
            log_entry = {
                'received_at': datetime.now().isoformat(),
                'signal': signal
            }
            
            # Load existing logs
            logs = []
            if os.path.exists(self.log_file):
                try:
                    with open(self.log_file, 'r') as f:
                        logs = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    logs = []
            
            # Add new log entry
            logs.append(log_entry)
            
            # Keep only last 1000 entries
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            # Write back to file
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error logging signal: {e}")
    
    async def _execute_trade(self, signal: dict):
        \"\"\"Execute trade based on signal\"\"\"
        try:
            symbol = signal['symbol']
            action = signal['action']
            details = signal.get('details', {})
            
            # Check if we should execute this trade
            confidence = details.get('confidence', 0)
            min_confidence = 0.7  # Configurable threshold
            
            if confidence < min_confidence:
                logger.info(f"Skipping trade for {symbol} - low confidence: {confidence:.3f}")
        except Exception as e:
            logger.error(f"Error in _execute_trade: {e}")
"""

# --------- CREATE STRUCTURE ---------

root = "BrittBot_Project"

# Top-level
write_file(f"{root}/.env", env_content)
write_file(f"{root}/.gitignore", gitignore_content)
write_file(f"{root}/README.md", readme_content)
write_file(f"{root}/run.py", runpy_content)

# AI
os.makedirs(f"{root}/BrittBot_AI/config", exist_ok=True)
write_file(f"{root}/BrittBot_AI/__init__.py", "")
write_file(f"{root}/BrittBot_AI/main.py", mainpy_content)
write_file(f"{root}/BrittBot_AI/ai_integration.py", aiintegrationpy_content)
write_file(f"{root}/BrittBot_AI/monitoring_utils.py", "# Monitoring and logging utils\n")
write_file(f"{root}/BrittBot_AI/requirements.txt", "# Add dependencies here\n")
write_file(f"{root}/BrittBot_AI/config/ai_config.json", "{}\n")
write_file(f"{root}/BrittBot_AI/config/rl_config.json", "{}\n")
write_file(f"{root}/BrittBot_AI/config/conservative_config.json", "{}\n")
write_file(f"{root}/BrittBot_AI/config/aggressive_config.json", "{}\n")

# Setup
write_file(f"{root}/BrittBot_Setup/__init__.py", "")
write_file(f"{root}/BrittBot_Setup/client.py", clientpy_content)
write_file(f"{root}/BrittBot_Setup/trading_client.py", "# Alpaca integration stub\n")
write_file(f"{root}/BrittBot_Setup/brittbot_signal_log.json", "[]\n")

print("✅ Project structure and code populated in ./BrittBot_Project/")