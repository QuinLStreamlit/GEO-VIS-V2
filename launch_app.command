#!/bin/bash
# One-click launcher for Mac
# Double-click this file to launch the Geotechnical Data Analysis Tool

echo "🚀 Starting Geotechnical Data Analysis Tool..."
echo "📍 Working directory: $(pwd)"

# Navigate to the script directory
cd "$(dirname "$0")"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit
fi

# Check if required packages are installed
echo "🔧 Checking dependencies..."
/opt/homebrew/bin/python3 -c "
import sys
required_packages = ['streamlit', 'pandas', 'numpy', 'matplotlib', 'plotly', 'openpyxl', 'watchdog', 'scipy', 'pyproj']
missing = []
for package in required_packages:
    try:
        __import__(package)
        print(f'✅ {package}')
    except ImportError:
        missing.append(package)
        print(f'❌ {package} - missing')

if missing:
    print(f'\n🔧 Installing missing packages: {missing}')
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--break-system-packages'] + missing)
    print('✅ All packages installed!')
"

# Install Xcode command line tools for watchdog if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🔧 Checking Xcode command line tools for better performance..."
    if ! xcode-select -p &> /dev/null; then
        echo "📦 Installing Xcode command line tools for optimal performance..."
        xcode-select --install 2>/dev/null || echo "⚠️  Xcode tools installation may require manual approval"
    else
        echo "✅ Xcode command line tools already installed"
    fi
fi

# Kill any existing streamlit processes
echo "🧹 Cleaning up existing processes..."
pkill -f streamlit 2>/dev/null || true

# Wait a moment
sleep 2

# Launch the application
echo "🎯 Launching application..."
echo "📂 Your browser will open automatically at: http://localhost:8501"
echo "🔄 To stop the application, close this terminal or press Ctrl+C"
echo ""

# Try to open browser automatically
sleep 3 && open http://localhost:8501 &

# Run streamlit with anaconda Python to ensure pyproj availability - disable auto browser
/opt/anaconda3/bin/python -m streamlit run main_app.py --server.port=8501 --browser.gatherUsageStats=false --server.headless=true

echo "👋 Application closed. Thank you for using the Geotechnical Data Analysis Tool!"
read -p "Press Enter to close this window..."