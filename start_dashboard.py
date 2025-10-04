#!/usr/bin/env python3
"""
Healthcare Dashboard Startup Script
Automated setup and startup for the healthcare dashboard system
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

class DashboardStarter:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.server_process = None
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        if sys.version_info < (3, 8):
            print("❌ Python 3.8 or higher is required")
            print(f"Current version: {sys.version}")
            return False
        print(f"✅ Python version: {sys.version.split()[0]}")
        return True
    
    def install_dependencies(self):
        """Install required dependencies"""
        print("\n📦 Installing dependencies...")
        requirements_file = self.base_dir / "requirements.txt"
        
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True, capture_output=True, text=True)
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("Try running manually: pip install -r requirements.txt")
            return False
    
    def setup_databases(self):
        """Initialize databases if needed"""
        print("\n🗄️  Setting up databases...")
        
        # Check if databases exist
        users_db = self.base_dir / "users.db"
        memories_db = self.base_dir / "memories.db"
        
        if users_db.exists():
            print("✅ users.db found")
        else:
            print("ℹ️  users.db will be created on first run")
        
        if memories_db.exists():
            print("✅ memories.db found")
        else:
            print("ℹ️  memories.db will be created on first run")
        
        return True
    
    def create_env_file(self):
        """Create a basic .env file if it doesn't exist"""
        env_file = self.base_dir / ".env"
        
        if env_file.exists():
            print("✅ .env file found")
            return True
        
        print("\n📝 Creating basic .env file...")
        env_content = """# Healthcare Dashboard Environment Configuration
# Optional: Add your API keys here

# Gemini AI API Key (for enhanced insights)
# GEMINI_API_KEY=your_gemini_api_key_here

# Hugging Face API Key (for AI art)
# HF_API_KEY=your_huggingface_api_key_here

# Security (change in production)
SECRET_KEY=healthcare_dashboard_secret_key_change_in_production

# Debug mode
DEBUG=True
"""
        
        try:
            with open(env_file, 'w') as f:
                f.write(env_content)
            print("✅ Created .env file with default settings")
            return True
        except Exception as e:
            print(f"⚠️  Could not create .env file: {e}")
            return True  # Not critical
    
    def start_server(self):
        """Start the FastAPI server"""
        print("\n🚀 Starting Healthcare Dashboard server...")
        
        main_file = self.base_dir / "main.py"
        if not main_file.exists():
            print("❌ main.py not found")
            return False
        
        try:
            # Start the server
            self.server_process = subprocess.Popen([
                sys.executable, str(main_file)
            ], cwd=str(self.base_dir))
            
            print("✅ Server starting...")
            print("📍 Server will be available at: http://localhost:8000")
            
            # Wait a moment for server to start
            print("⏳ Waiting for server to initialize...")
            time.sleep(5)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    def open_browser(self):
        """Open the dashboard in the default browser"""
        print("\n🌐 Opening dashboard in browser...")
        
        urls_to_open = [
            "http://localhost:8000/dashboard",
            "http://localhost:8000"
        ]
        
        for url in urls_to_open:
            try:
                webbrowser.open(url)
                print(f"✅ Opened {url}")
                break
            except Exception as e:
                print(f"⚠️  Could not open {url}: {e}")
    
    def show_usage_instructions(self):
        """Show usage instructions"""
        print("\n" + "="*60)
        print("🏥 HEALTHCARE DASHBOARD - READY TO USE!")
        print("="*60)
        print("\n📋 Available URLs:")
        print("• Main Dashboard: http://localhost:8000/dashboard")
        print("• Patient Interface: http://localhost:8000/patient")
        print("• API Documentation: http://localhost:8000/docs")
        print("• Health Check: http://localhost:8000/health")
        
        print("\n🧪 Testing Real-Time Features:")
        print("1. Open dashboard in one browser tab")
        print("2. Open patient interface in another tab")
        print("3. Interact with games, chat, or other features")
        print("4. Watch real-time updates in the dashboard!")
        
        print("\n🔧 Development Commands:")
        print("• Run tests: pytest tests/test_realtime_dashboard.py -v")
        print("• Demo script: python demo_realtime.py")
        print("• Validate setup: python validate_setup.py")
        
        print("\n⚠️  To stop the server: Press Ctrl+C")
        print("="*60)
    
    def cleanup(self):
        """Clean up resources"""
        if self.server_process:
            print("\n🛑 Stopping server...")
            self.server_process.terminate()
            self.server_process.wait()
    
    def run(self):
        """Run the complete startup sequence"""
        print("🏥 Healthcare Dashboard Startup")
        print("="*40)
        
        try:
            # Pre-flight checks
            if not self.check_python_version():
                return False
            
            # Setup steps
            if not self.install_dependencies():
                return False
            
            if not self.setup_databases():
                return False
            
            self.create_env_file()
            
            # Start the server
            if not self.start_server():
                return False
            
            # Open browser
            self.open_browser()
            
            # Show instructions
            self.show_usage_instructions()
            
            # Keep running
            print("\n⏳ Server is running. Press Ctrl+C to stop...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n👋 Shutting down...")
                return True
                
        except Exception as e:
            print(f"\n❌ Startup failed: {e}")
            return False
        finally:
            self.cleanup()

def main():
    starter = DashboardStarter()
    success = starter.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
