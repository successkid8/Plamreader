# 🔮 Palmora - AI-Powered Palm Reading

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

Professional AI-powered palm reading application with step-by-step guidance and beautiful visualizations.

## ✨ Features

- 📱 **Mobile-Responsive Design** - Works perfectly on all devices
- 🤖 **Advanced AI Analysis** - Powered by OpenAI GPT-4 Vision
- 📷 **Smart Image Capture** - Camera + upload with quality validation
- 🎨 **Beautiful Visualizations** - AI-generated palm art and reports
- 📄 **PDF Reports** - Downloadable comprehensive readings
- 🔧 **Image Enhancement** - Auto-optimization for best results

## 🚀 Live Demo

Visit the live app: [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)

## 📱 Mobile-Optimized

This app is fully optimized for mobile devices with:
- Touch-friendly interface (44px+ touch targets)
- Responsive layouts for all screen sizes
- Mobile-specific camera integration
- Swipe-friendly navigation

## 🛠 Tech Stack

- **Frontend:** Streamlit with custom CSS
- **AI Models:** OpenAI GPT-4o Vision, DALL-E 3
- **Image Processing:** Pillow, OpenCV
- **PDF Generation:** ReportLab
- **Deployment:** Streamlit Cloud

## ⚙️ Configuration

### Required Environment Variables

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Optional Configuration

```bash
PALM_READER_VISION_MODEL=gpt-4o
PALM_READER_IMAGE_MODEL=dall-e-3
```

## 🏗 Local Development

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set Environment Variables:**
```bash
export OPENAI_API_KEY="your-api-key"
```

3. **Run the App:**
```bash
streamlit run app.py
```

## 🌐 Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Connect repository at [share.streamlit.io](https://share.streamlit.io)
3. Add secrets in the Streamlit Cloud dashboard
4. Deploy automatically

### Docker Deployment

```bash
docker build -t palmora .
docker run -p 8501:8501 palmora
```

## 📋 Requirements

- Python 3.11+
- OpenAI API key with GPT-4 Vision access
- Internet connection for AI processing

## 🔒 Privacy & Security

- Images processed in memory only
- No persistent storage of palm photos
- API keys handled securely via Streamlit secrets
- Entertainment purposes only

## 📞 Support

For issues and feature requests, please create an issue in the repository.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for the future of palm reading technology**