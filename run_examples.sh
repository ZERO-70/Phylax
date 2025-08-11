#!/bin/bash
# Usage Examples for Your Enhanced Abuse Detection System

echo "🚀 Abuse Detection System - Usage Guide"
echo "======================================="
echo ""

# Activate environment
echo "📁 Activating environment..."
source enviroment/bin/activate

echo ""
echo "🎯 Available Commands:"
echo ""

echo "1. 📺 Full Detection with Video Display:"
echo "   python receiver.py --enable-nsfw --no-gun --enable-transcription --enable-profanity"
echo ""

echo "2. 🖥️  Headless Mode (No Video Display - Your Request):"
echo "   python receiver.py --headless --enable-nsfw --no-gun --enable-transcription --enable-profanity"
echo ""

echo "3. 🔍 Test GPU Detection:"
echo "   python test_gpu.py"
echo ""

echo "4. 📊 View Live Transcript Logs:"
echo "   tail -f audio_transcript.txt"
echo ""

echo "5. 🧪 Test Headless Mode:"
echo "   python test_headless.py"
echo ""

echo "📝 Key Features Implemented:"
echo "  ✅ Automatic GPU detection (CPU fallback working)"
echo "  ✅ Headless mode for SSH/remote usage"  
echo "  ✅ Temperature logging in audio_transcript.txt"
echo "  ✅ NSFW detection with Falconsai model"
echo "  ✅ Real-time audio transcription"
echo "  ✅ Profanity filtering"
echo "  ✅ Enhanced logging and monitoring"
echo ""

echo "🎮 Ready to run! Choose a command above or run with --help for more options."
