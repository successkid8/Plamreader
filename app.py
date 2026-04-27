from __future__ import annotations

import html
import io
import os
import re
import time
from datetime import datetime
from typing import Optional, Tuple

import streamlit as st
from openai import OpenAI
from PIL import Image, ImageEnhance, ImageFilter

from palm_reader import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_VISION_MODEL,
    PalmReading,
    create_pdf_bytes,
    generate_contour_image,
    generate_report,
    optimize_upload_image,
    split_report_sections,
    validate_palm_photo,
)


st.set_page_config(
    page_title="Palmora - AI Palm Reading",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap');
    
    :root {
        /* Professional Product Color Palette */
        --primary-900: #1f2937;
        --primary-800: #374151;
        --primary-700: #4b5563;
        --primary-600: #6b7280;
        --primary-500: #9ca3af;
        --primary-400: #d1d5db;
        --primary-300: #e5e7eb;
        --primary-200: #f3f4f6;
        --primary-100: #f9fafb;
        --primary-50: #ffffff;
        
        /* Professional Accent Colors */
        --accent-emerald: #059669;
        --accent-purple: #7c3aed;
        --accent-orange: #ea580c;
        --accent-pink: #db2777;
        --accent-cyan: #0891b2;
        
        /* Professional Neutral Palette */
        --neutral-950: #1f2937;
        --neutral-900: #374151;
        --neutral-800: #4b5563;
        --neutral-700: #6b7280;
        --neutral-600: #9ca3af;
        --neutral-500: #d1d5db;
        --neutral-400: #e5e7eb;
        --neutral-300: #f3f4f6;
        --neutral-200: #f9fafb;
        --neutral-100: #ffffff;
        --neutral-50: #ffffff;
        
        /* Professional Semantic Colors */
        --success: #059669;
        --warning: #ea580c;
        --error: #dc2626;
        --info: #0891b2;
        
        /* Professional Glass Effects */
        --glass-bg: rgba(255, 255, 255, 0.95);
        --glass-border: rgba(229, 231, 235, 0.8);
        --glass-shadow: 0 4px 16px rgba(107, 114, 128, 0.15);
        
        /* Modern Shadows */
        --shadow-xs: 0 1px 2px rgba(0, 0, 0, 0.05);
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06);
        --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1), 0 4px 6px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px rgba(0, 0, 0, 0.1), 0 10px 10px rgba(0, 0, 0, 0.04);
        --shadow-2xl: 0 25px 50px rgba(0, 0, 0, 0.25);
        
        /* Professional Gradients */
        --gradient-primary: linear-gradient(135deg, var(--neutral-900) 0%, var(--neutral-700) 100%);
        --gradient-accent: linear-gradient(135deg, var(--accent-emerald) 0%, var(--accent-cyan) 100%);
        --gradient-warm: linear-gradient(135deg, var(--accent-orange) 0%, var(--accent-pink) 100%);
        --gradient-cool: linear-gradient(135deg, var(--neutral-700) 0%, var(--accent-purple) 100%);
        --gradient-surface: linear-gradient(145deg, #ffffff 0%, #f9fafb 50%, #f3f4f6 100%);
        
        /* Border Radius */
        --radius-xs: 0.25rem;
        --radius-sm: 0.375rem;
        --radius-md: 0.5rem;
        --radius-lg: 0.75rem;
        --radius-xl: 1rem;
        --radius-2xl: 1.5rem;
        --radius-3xl: 2rem;
        --radius-full: 9999px;
        
        /* Spacing Scale */
        --space-px: 1px;
        --space-0: 0;
        --space-1: 0.25rem;
        --space-2: 0.5rem;
        --space-3: 0.75rem;
        --space-4: 1rem;
        --space-5: 1.25rem;
        --space-6: 1.5rem;
        --space-8: 2rem;
        --space-10: 2.5rem;
        --space-12: 3rem;
        --space-16: 4rem;
        --space-20: 5rem;
        --space-24: 6rem;
        --space-32: 8rem;
    }
    
    /* Base Styles */
    * {
        box-sizing: border-box;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f9fafb 0%, #ffffff 50%, #f3f4f6 100%);
        background-attachment: fixed;
        color: var(--neutral-900);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        line-height: 1.6;
    }
    
    /* Container & Layout */
    .block-container {
        padding: var(--space-4) var(--space-4) var(--space-8);
        max-width: 1280px;
        margin: 0 auto;
    }
    
    /* Mobile-First Responsive Design */
    @media (max-width: 480px) {
        .block-container {
            padding: var(--space-2) var(--space-2) var(--space-4);
        }
    }
    
    /* Typography System */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        line-height: 1.2;
        letter-spacing: -0.02em;
        margin: 0;
        color: var(--neutral-900);
    }
    
    h1 {
        font-size: clamp(1.8rem, 6vw, 4rem);
        font-weight: 800;
        color: var(--neutral-900);
        margin-bottom: var(--space-4);
        line-height: 1.1;
    }
    
    /* Mobile Typography */
    @media (max-width: 480px) {
        h1 {
            font-size: clamp(1.5rem, 8vw, 2.5rem);
            line-height: 1.2;
            margin-bottom: var(--space-3);
        }
    }
    
    h2 {
        font-size: clamp(1.5rem, 4vw, 2.5rem);
        font-weight: 700;
        margin-bottom: var(--space-3);
        line-height: 1.2;
    }
    
    /* Mobile H2 */
    @media (max-width: 480px) {
        h2 {
            font-size: clamp(1.25rem, 6vw, 2rem);
            margin-bottom: var(--space-2);
        }
    }
    
    h3 {
        font-size: clamp(1.25rem, 3vw, 1.75rem);
        font-weight: 600;
        margin-bottom: var(--space-2);
    }
    
    /* Professional Brand Header */
    .brand-header {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-xl);
        padding: var(--space-6);
        margin-bottom: var(--space-8);
        box-shadow: var(--glass-shadow);
        position: relative;
        overflow: hidden;
    }
    
    .brand-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--accent-emerald);
    }
    
    /* Mobile Header Optimization */
    @media (max-width: 768px) {
        .brand-header {
            padding: var(--space-4);
            margin-bottom: var(--space-4);
            border-radius: var(--radius-lg);
        }
    }
    
    @media (max-width: 480px) {
        .brand-header {
            padding: var(--space-3);
            margin-bottom: var(--space-3);
        }
    }
    
    .brand-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: var(--space-6);
    }
    
    /* Mobile Navigation */
    @media (max-width: 640px) {
        .brand-nav {
            flex-direction: column;
            gap: var(--space-3);
            text-align: center;
            margin-bottom: var(--space-4);
        }
    }
    
    .brand-logo {
        display: flex;
        align-items: center;
        gap: var(--space-3);
    }
    
    .logo-icon {
        width: 48px;
        height: 48px;
        background: var(--accent-emerald);
        border-radius: var(--radius-lg);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .logo-icon::before {
        content: '🔮';
        position: relative;
        z-index: 2;
    }
    
    .logo-icon:hover {
        background: var(--accent-cyan);
        transform: scale(1.05);
    }
    
    .brand-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--neutral-900);
        margin: 0;
    }
    
    .brand-badge {
        background: var(--neutral-900);
        color: white;
        padding: var(--space-2) var(--space-4);
        border-radius: var(--radius-full);
        font-size: 0.875rem;
        font-weight: 600;
        box-shadow: var(--shadow-sm);
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: var(--space-8) 0;
    }
    
    .hero-subtitle {
        font-size: clamp(1.125rem, 2.5vw, 1.375rem);
        color: var(--neutral-600);
        max-width: 600px;
        margin: 0 auto var(--space-8);
        line-height: 1.6;
    }
    
    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: var(--space-6);
        margin: var(--space-8) 0;
    }
    
    /* Mobile Feature Grid */
    @media (max-width: 768px) {
        .feature-grid {
            grid-template-columns: 1fr;
            gap: var(--space-4);
            margin: var(--space-4) 0;
        }
    }
    
    @media (max-width: 480px) {
        .feature-grid {
            gap: var(--space-3);
            margin: var(--space-3) 0;
        }
    }
    
    .feature-card {
        background: white;
        border: 1px solid var(--neutral-300);
        border-radius: var(--radius-xl);
        padding: var(--space-6);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    
    /* Mobile Feature Cards */
    @media (max-width: 768px) {
        .feature-card {
            padding: var(--space-4);
            border-radius: var(--radius-lg);
        }
    }
    
    @media (max-width: 480px) {
        .feature-card {
            padding: var(--space-3);
            text-align: center;
        }
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: var(--accent-emerald);
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover::before {
        transform: scaleX(1);
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
        border-color: var(--neutral-400);
    }
    
    .feature-icon {
        width: 56px;
        height: 56px;
        background: var(--accent-emerald);
        border-radius: var(--radius-lg);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: var(--space-4);
        box-shadow: var(--shadow-sm);
    }
    
    .feature-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--neutral-900);
        margin-bottom: var(--space-2);
    }
    
    .feature-description {
        color: var(--neutral-600);
        line-height: 1.6;
    }
    
    /* Professional Capture Interface */
    .capture-container {
        background: white;
        border: 1px solid var(--neutral-300);
        border-radius: var(--radius-xl);
        padding: var(--space-8);
        margin: var(--space-8) 0;
        box-shadow: var(--shadow-md);
    }
    
    .capture-tabs {
        margin-bottom: var(--space-6);
    }
    
    /* Professional Photo Guide */
    .photo-guide {
        background: var(--neutral-100);
        border: 1px solid var(--neutral-300);
        border-radius: var(--radius-xl);
        padding: var(--space-6);
        margin: var(--space-6) 0;
    }
    
    .guide-title {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        font-weight: 700;
        color: var(--neutral-900);
        margin-bottom: var(--space-4);
    }
    
    .guide-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .guide-item {
        display: flex;
        align-items: center;
        gap: var(--space-3);
        padding: var(--space-2) 0;
        color: var(--neutral-700);
    }
    
    .guide-item::before {
        content: '✓';
        background: var(--accent-emerald);
        color: white;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: bold;
        flex-shrink: 0;
    }
    
    /* Cards & Reports */
    .app-card, .report-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: var(--radius-xl);
        padding: var(--space-6);
        margin-bottom: var(--space-4);
        box-shadow: var(--shadow-lg);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .app-card::before, .report-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: var(--gradient-accent);
    }
    
    .app-card:hover, .report-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-2xl);
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--neutral-900);
        margin-bottom: var(--space-3);
        padding-left: var(--space-4);
    }
    
    .card-body {
        color: var(--neutral-700);
        line-height: 1.7;
        padding-left: var(--space-4);
    }
    
    /* Media Grid */
    .result-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: var(--space-6);
        margin: var(--space-6) 0;
    }
    
    .media-card {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: var(--radius-xl);
        padding: var(--space-4);
        box-shadow: var(--shadow-md);
        transition: transform 0.3s ease;
    }
    
    .media-card:hover {
        transform: scale(1.02);
    }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: var(--space-4) 0;
        margin: var(--space-8) 0 var(--space-4);
        border-bottom: 2px solid var(--neutral-200);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--neutral-900);
        margin: 0;
    }
    
    .section-badge {
        background: var(--neutral-800);
        color: white;
        padding: var(--space-2) var(--space-3);
        border-radius: var(--radius-full);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Professional Privacy Notice */
    .privacy-notice {
        background: var(--neutral-100);
        border: 1px solid var(--neutral-400);
        border-radius: var(--radius-lg);
        padding: var(--space-4);
        margin: var(--space-4) 0;
        color: var(--neutral-700);
    }
    
    /* Animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Streamlit Overrides */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--neutral-200);
        border-radius: var(--radius-lg);
        padding: var(--space-1);
        gap: var(--space-1);
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }
    
    /* Mobile Tabs */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab-list"] {
            flex-wrap: nowrap;
            justify-content: flex-start;
        }
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: var(--radius-md);
        transition: all 0.3s ease;
        font-weight: 600;
        color: var(--neutral-700);
        min-height: 44px;
        padding: var(--space-2) var(--space-3);
    }
    
    /* Mobile Tab Buttons */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab"] {
            min-height: 48px;
            padding: var(--space-3) var(--space-4);
            font-size: 0.9rem;
            white-space: nowrap;
            flex-shrink: 0;
        }
    }
    
    @media (max-width: 480px) {
        .stTabs [data-baseweb="tab"] {
            min-height: 52px;
            font-size: 0.85rem;
        }
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: white;
        box-shadow: var(--shadow-sm);
        color: var(--neutral-900);
    }
    
    .stButton > button {
        background: var(--accent-emerald);
        color: white;
        border: none;
        border-radius: var(--radius-lg);
        padding: var(--space-3) var(--space-6);
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
        min-height: 44px; /* Touch-friendly minimum */
        font-size: 1rem;
    }
    
    /* Mobile Button Optimization */
    @media (max-width: 768px) {
        .stButton > button {
            min-height: 48px;
            padding: var(--space-4) var(--space-6);
            font-size: 1.1rem;
            font-weight: 600;
        }
    }
    
    @media (max-width: 480px) {
        .stButton > button {
            min-height: 52px;
            padding: var(--space-4) var(--space-8);
            font-size: 1.2rem;
            border-radius: var(--radius-xl);
        }
    }
    
    .stButton > button:hover {
        background: var(--accent-cyan);
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    
    .stDownloadButton > button {
        background: var(--neutral-700);
        color: white;
        border: none;
        border-radius: var(--radius-lg);
        padding: var(--space-3) var(--space-4);
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .stDownloadButton > button:hover {
        background: var(--neutral-600);
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .block-container {
            padding: var(--space-3);
        }
        
        .brand-nav {
            flex-direction: column;
            gap: var(--space-4);
            text-align: center;
        }
        
        .brand-badge {
            order: -1;
        }
        
        .feature-grid {
            grid-template-columns: 1fr;
            gap: var(--space-4);
        }
        
        .result-grid {
            grid-template-columns: 1fr;
        }
        
        .section-header {
            flex-direction: column;
            align-items: flex-start;
            gap: var(--space-2);
        }
        
        .capture-container {
            padding: var(--space-4);
        }
    }
    
    /* Accessibility */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    
    /* Focus States */
    *:focus-visible {
        outline: 2px solid var(--accent-emerald);
        outline-offset: 2px;
        border-radius: var(--radius-sm);
    }
    
    /* Step Wizard Interface */
    .step-progress {
        background: white;
        border: 1px solid var(--neutral-300);
        border-radius: var(--radius-xl);
        padding: var(--space-6);
        margin: var(--space-6) 0;
        box-shadow: var(--shadow-sm);
    }
    
    /* Mobile Step Progress */
    @media (max-width: 768px) {
        .step-progress {
            padding: var(--space-4);
            margin: var(--space-4) 0;
            border-radius: var(--radius-lg);
        }
    }
    
    @media (max-width: 480px) {
        .step-progress {
            padding: var(--space-3);
            margin: var(--space-3) 0;
        }
    }
    
    .progress-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-4);
    }
    
    /* Mobile Progress Header */
    @media (max-width: 640px) {
        .progress-header {
            flex-direction: column;
            gap: var(--space-2);
            text-align: center;
        }
    }
    
    .progress-header h3 {
        margin: 0;
        color: var(--neutral-900);
        font-size: 1.25rem;
    }
    
    .step-counter {
        background: var(--accent-emerald);
        color: white;
        padding: var(--space-1) var(--space-3);
        border-radius: var(--radius-full);
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .progress-bar {
        width: 100%;
        height: 8px;
        background: var(--neutral-200);
        border-radius: var(--radius-full);
        overflow: hidden;
        margin-bottom: var(--space-4);
    }
    
    .progress-fill {
        height: 100%;
        background: var(--gradient-accent);
        border-radius: var(--radius-full);
        transition: width 0.5s ease;
    }
    
    .step-labels {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: var(--space-2);
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* Mobile Step Labels */
    @media (max-width: 640px) {
        .step-labels {
            grid-template-columns: repeat(2, 1fr);
            gap: var(--space-3);
            font-size: 0.8rem;
        }
    }
    
    @media (max-width: 480px) {
        .step-labels {
            font-size: 0.75rem;
        }
    }
    
    .step-labels span {
        text-align: center;
        color: var(--neutral-500);
        transition: color 0.3s ease;
    }
    
    .step-labels span.active {
        color: var(--accent-emerald);
        font-weight: 600;
    }
    
    /* Step Cards */
    .step-card {
        background: white;
        border: 2px solid var(--accent-emerald);
        border-radius: var(--radius-xl);
        padding: var(--space-6);
        margin: var(--space-6) 0;
        box-shadow: var(--shadow-lg);
    }
    
    /* Mobile Step Cards */
    @media (max-width: 768px) {
        .step-card {
            padding: var(--space-4);
            margin: var(--space-4) 0;
            border-radius: var(--radius-lg);
            border-width: 1px;
        }
    }
    
    @media (max-width: 480px) {
        .step-card {
            padding: var(--space-3);
            margin: var(--space-3) 0;
        }
    }
    
    .step-header {
        display: flex;
        align-items: center;
        gap: var(--space-4);
        margin-bottom: var(--space-4);
    }
    
    /* Mobile Step Headers */
    @media (max-width: 640px) {
        .step-header {
            flex-direction: column;
            text-align: center;
            gap: var(--space-3);
        }
    }
    
    .step-icon {
        width: 64px;
        height: 64px;
        background: var(--accent-emerald);
        border-radius: var(--radius-xl);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        color: white;
        flex-shrink: 0;
    }
    
    /* Mobile Step Icons */
    @media (max-width: 768px) {
        .step-icon {
            width: 56px;
            height: 56px;
            font-size: 1.75rem;
            border-radius: var(--radius-lg);
        }
    }
    
    @media (max-width: 480px) {
        .step-icon {
            width: 48px;
            height: 48px;
            font-size: 1.5rem;
        }
    }
    
    .step-header h2 {
        margin: 0;
        color: var(--neutral-900);
        font-size: 1.75rem;
    }
    
    .step-header p {
        margin: 0;
        color: var(--neutral-600);
        font-size: 1rem;
    }
    
    /* Navigation Buttons */
    .stButton > button[kind="secondary"] {
        background: var(--neutral-200);
        color: var(--neutral-700);
        border: 1px solid var(--neutral-300);
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: var(--neutral-300);
        color: var(--neutral-900);
    }
    
    /* Enhanced Radio Buttons */
    .stRadio > div {
        display: flex;
        gap: var(--space-4);
        flex-wrap: wrap;
    }
    
    .stRadio > div > label {
        background: white;
        border: 2px solid var(--neutral-300);
        border-radius: var(--radius-lg);
        padding: var(--space-3) var(--space-4);
        cursor: pointer;
        transition: all 0.3s ease;
        min-width: 150px;
        text-align: center;
        min-height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Mobile Radio Buttons */
    @media (max-width: 768px) {
        .stRadio > div {
            flex-direction: column;
            gap: var(--space-3);
        }
        
        .stRadio > div > label {
            min-width: auto;
            width: 100%;
            min-height: 48px;
            font-size: 1rem;
            padding: var(--space-4);
        }
    }
    
    @media (max-width: 480px) {
        .stRadio > div > label {
            min-height: 52px;
            font-size: 1.1rem;
            border-radius: var(--radius-xl);
        }
    }
    
    .stRadio > div > label:hover {
        border-color: var(--accent-emerald);
        background: var(--neutral-50);
    }
    
    .stRadio > div > label[data-checked="true"] {
        border-color: var(--accent-emerald);
        background: var(--accent-emerald);
        color: white;
    }
    
    /* Results Page Styling */
    .results-hero {
        background: linear-gradient(135deg, var(--accent-emerald) 0%, var(--accent-cyan) 100%);
        color: white;
        border-radius: var(--radius-xl);
        padding: var(--space-8);
        margin: var(--space-6) 0;
        box-shadow: var(--shadow-xl);
    }
    
    .results-header {
        display: flex;
        align-items: center;
        gap: var(--space-4);
        justify-content: space-between;
    }
    
    /* Mobile Results Header */
    @media (max-width: 768px) {
        .results-header {
            flex-direction: column;
            text-align: center;
            gap: var(--space-4);
        }
    }
    
    .results-icon {
        width: 80px;
        height: 80px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: var(--radius-xl);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        flex-shrink: 0;
    }
    
    /* Mobile Results Icon */
    @media (max-width: 768px) {
        .results-icon {
            width: 64px;
            height: 64px;
            font-size: 2rem;
            border-radius: var(--radius-lg);
        }
    }
    
    @media (max-width: 480px) {
        .results-icon {
            width: 56px;
            height: 56px;
            font-size: 1.75rem;
        }
    }
    
    .results-content h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
    }
    
    .results-content p {
        margin: 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    .results-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: var(--space-2) var(--space-4);
        border-radius: var(--radius-full);
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    /* Summary Cards */
    .summary-card {
        background: white;
        border: 1px solid var(--neutral-300);
        border-radius: var(--radius-xl);
        padding: var(--space-6);
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
        height: 100%;
    }
    
    .summary-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
        border-color: var(--accent-emerald);
    }
    
    .summary-card .card-icon {
        font-size: 2.5rem;
        margin-bottom: var(--space-3);
    }
    
    .summary-card h3 {
        color: var(--neutral-900);
        margin: 0 0 var(--space-2);
        font-size: 1.25rem;
    }
    
    .summary-card p {
        color: var(--neutral-600);
        margin: 0;
        line-height: 1.5;
    }
    
    /* Action Section */
    .action-section {
        background: var(--neutral-100);
        border-radius: var(--radius-xl);
        padding: var(--space-6);
        text-align: center;
        margin: var(--space-8) 0 var(--space-4);
    }
    
    .action-section h2 {
        margin: 0 0 var(--space-2);
        color: var(--neutral-900);
    }
    
    .action-section p {
        margin: 0;
        color: var(--neutral-600);
    }
    
    /* Action Cards */
    .action-card {
        background: white;
        border: 2px solid var(--neutral-300);
        border-radius: var(--radius-xl);
        padding: var(--space-4);
        text-align: center;
        margin-bottom: var(--space-3);
        transition: all 0.3s ease;
    }
    
    .action-card:hover {
        border-color: var(--accent-emerald);
        background: var(--neutral-50);
    }
    
    .action-card .action-icon {
        font-size: 2rem;
        margin-bottom: var(--space-2);
    }
    
    .action-card h3 {
        margin: 0 0 var(--space-1);
        color: var(--neutral-900);
        font-size: 1.1rem;
    }
    
    .action-card p {
        margin: 0;
        color: var(--neutral-600);
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    /* Enhanced Download Button */
    .stDownloadButton > button {
        background: var(--gradient-accent) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-lg) !important;
        padding: var(--space-3) var(--space-4) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--shadow-md) !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-lg) !important;
    }
    
    /* Feedback Buttons */
    .stButton > button[kind="secondary"] {
        background: white;
        color: var(--neutral-700);
        border: 2px solid var(--neutral-300);
        transition: all 0.3s ease;
    }
    
    .stButton > button[kind="secondary"]:hover {
        border-color: var(--accent-emerald);
        color: var(--accent-emerald);
        background: var(--neutral-50);
    }
    
    /* Responsive Adjustments */
    @media (max-width: 768px) {
        .step-labels {
            grid-template-columns: repeat(2, 1fr);
            gap: var(--space-3);
        }
        
        .step-header {
            flex-direction: column;
            text-align: center;
        }
        
        .step-icon {
            width: 56px;
            height: 56px;
            font-size: 1.75rem;
        }
        
        .progress-header {
            flex-direction: column;
            gap: var(--space-2);
        }
        
        .results-header {
            flex-direction: column;
            text-align: center;
            gap: var(--space-4);
        }
        
        .results-content h1 {
            font-size: 1.75rem;
        }
        
        .results-icon {
            width: 64px;
            height: 64px;
            font-size: 2rem;
        }
    }
    
    /* Professional Upload Interface */
    .stFileUploader > div {
        border: 2px dashed var(--neutral-400);
        border-radius: var(--radius-xl);
        padding: var(--space-8);
        background: white;
        transition: all 0.3s ease;
        min-height: 150px;
    }
    
    /* Mobile Upload Interface */
    @media (max-width: 768px) {
        .stFileUploader > div {
            padding: var(--space-6);
            border-radius: var(--radius-lg);
            min-height: 120px;
        }
    }
    
    @media (max-width: 480px) {
        .stFileUploader > div {
            padding: var(--space-4);
            min-height: 100px;
        }
    }
    
    .stFileUploader > div:hover {
        border-color: var(--accent-emerald);
        background: var(--neutral-50);
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    
    .stFileUploader > div[data-testid="stFileUploaderDropzone"] {
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    
    /* Mobile File Uploader */
    @media (max-width: 768px) {
        .stFileUploader > div[data-testid="stFileUploaderDropzone"] {
            min-height: 150px;
        }
    }
    
    @media (max-width: 480px) {
        .stFileUploader > div[data-testid="stFileUploaderDropzone"] {
            min-height: 120px;
        }
    }
    
    /* Mobile Checkbox Enhancement */
    .stCheckbox {
        font-size: 1rem;
    }
    
    @media (max-width: 768px) {
        .stCheckbox {
            font-size: 1.1rem;
        }
        
        .stCheckbox > label {
            min-height: 44px;
            display: flex;
            align-items: center;
        }
    }
    
    @media (max-width: 480px) {
        .stCheckbox {
            font-size: 1.2rem;
        }
        
        .stCheckbox > label {
            min-height: 48px;
        }
    }
    
    /* Professional Metrics */
    .metric-container {
        background: white;
        border: 1px solid var(--neutral-300);
        border-radius: var(--radius-lg);
        padding: var(--space-4);
        transition: transform 0.3s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .metric-container:hover {
        transform: scale(1.01);
        box-shadow: var(--shadow-md);
    }
    
    /* Professional Progress and Status */
    .stProgress > div > div {
        background: var(--accent-emerald);
        border-radius: var(--radius-full);
    }
    
    .stStatus {
        background: white;
        border: 1px solid var(--neutral-300);
        border-radius: var(--radius-xl);
        box-shadow: var(--shadow-sm);
    }
    
    /* Professional Alerts */
    .stSuccess {
        background: #f0fdf4;
        border: 1px solid var(--success);
        border-radius: var(--radius-lg);
    }
    
    .stError {
        background: #fef2f2;
        border: 1px solid var(--error);
        border-radius: var(--radius-lg);
    }
    
    .stWarning {
        background: #fffbeb;
        border: 1px solid var(--warning);
        border-radius: var(--radius-lg);
    }
    
    .stInfo {
        background: #f0f9ff;
        border: 1px solid var(--info);
        border-radius: var(--radius-lg);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def validate_uploaded_image(uploaded_file) -> Tuple[bool, str, Optional[dict]]:
    """Validate uploaded image file and return quality metrics."""
    try:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Basic validation
        if image.width < 400 or image.height < 400:
            return False, "Image resolution too low. Minimum 400x400 pixels required.", None
        
        if uploaded_file.size > 10 * 1024 * 1024:  # 10MB
            return False, "File size too large. Maximum 10MB allowed.", None
        
        # Quality assessment
        resolution_score = min(100, (image.width * image.height) // 5000)
        aspect_ratio = image.width / image.height
        aspect_score = 100 if 0.7 <= aspect_ratio <= 1.4 else 50
        
        # Calculate overall quality
        quality_score = (resolution_score * 0.6) + (aspect_score * 0.4)
        
        metrics = {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode,
            "size": uploaded_file.size,
            "quality_score": int(quality_score),
            "resolution_score": int(resolution_score),
            "aspect_score": int(aspect_score)
        }
        
        return True, "Image validation successful", metrics
        
    except Exception as e:
        return False, f"Error processing image: {str(e)}", None


def enhance_image(image_bytes: bytes, auto_enhance: bool = True, rotate: int = 0, 
                 crop_to_palm: bool = False, reduce_noise: bool = False) -> bytes:
    """Enhance uploaded image with various options."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Rotate if needed
        if rotate != 0:
            image = image.rotate(rotate, expand=True)
        
        # Auto enhancement
        if auto_enhance:
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Adjust brightness slightly
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.05)
        
        # Noise reduction
        if reduce_noise:
            image = image.filter(ImageFilter.MedianFilter(size=3))
        
        # Convert to RGB if needed
        if image.mode not in ['RGB', 'L']:
            image = image.convert('RGB')
        
        # Save enhanced image
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=95, optimize=True)
        return output.getvalue()
        
    except Exception as e:
        st.error(f"Error enhancing image: {str(e)}")
        return image_bytes


def get_client() -> OpenAI | None:
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        secret_key = None

    api_key = secret_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def render_card(title: str, body: str) -> None:
    # Import the formatting function
    from palm_reader import format_report_content
    
    # Clean the title of any HTML tags and emojis for display
    clean_title = re.sub(r'<[^>]+>', '', title)
    clean_title = html.escape(clean_title)
    
    # Format the body content properly
    if '<p>' in body or '<div>' in body or '<ul>' in body:
        # Already formatted HTML
        formatted_body = body
    else:
        # Convert markdown to HTML
        formatted_body = format_report_content(body)
    
    st.markdown(
        f"""
        <div class="app-card">
            <div class="card-title">{clean_title}</div>
            <div class="card-body">{formatted_body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_body(sections: dict[str, str], title: str, fallback: str = "This section will appear once the reading is generated.") -> str:
    for section_title, body in sections.items():
        if section_title.lower() == title.lower():
            return body
    return fallback


def matching_sections(sections: dict[str, str], keywords: tuple[str, ...]) -> dict[str, str]:
    matches: dict[str, str] = {}
    for title, body in sections.items():
        haystack = f"{title} {body}".lower()
        if any(keyword in haystack for keyword in keywords):
            matches[title] = body
    return matches


def render_header() -> None:
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-nav">
                <div class="brand-logo">
                    <div class="logo-icon"></div>
                    <h1 class="brand-title">Palmora</h1>
                </div>
                <div class="brand-badge">✨ AI-Powered Palm Reading</div>
            </div>
            <div class="hero-section">
                <h1>Discover Your Palm's Secrets</h1>
                <p class="hero-subtitle">
                    Experience the future of palm reading with advanced AI technology. 
                    Capture your palm and unlock personalized insights with beautiful visualizations, 
                    detailed analysis, and interactive reports.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="privacy-notice">
            <strong>🛡️ Privacy First:</strong> Your palm images are processed securely and temporarily. 
            This app is designed for entertainment purposes only and should not be used for identity verification or life-changing decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_privacy_note() -> None:
    with st.expander("Privacy notice"):
        st.markdown(
            """
            - We do not intentionally store palm photos in this app.
            - Uploaded images are sent to the configured AI provider only when you press **Read My Palm**.
            - For India deployments, treat palm images as sensitive personal data and publish a DPDP Act-aligned privacy notice covering purpose, consent, retention, grievance contact, and deletion requests.
            - Keep API keys server-side in Streamlit secrets or environment variables.
            """
        )


def render_photo_guide() -> None:
    st.markdown(
        """
        <div class="photo-guide">
            <div class="guide-title">
                <span>📷</span>
                <span>Perfect Palm Photo Tips</span>
            </div>
            <ul class="guide-list">
                <li class="guide-item">Open palm facing camera directly</li>
                <li class="guide-item">Fingers spread naturally</li>
                <li class="guide-item">Bright, even lighting</li>
                <li class="guide-item">Palm lines clearly visible</li>
                <li class="guide-item">Hand fills the frame</li>
                <li class="guide-item">Steady, blur-free shot</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_report(reading: PalmReading) -> None:
    sections = split_report_sections(reading.report_markdown)
    st.markdown(
        """
        <div class="section-header">
            <h2 class="section-title">🔮 Your Palmora Reading</h2>
            <div class="section-badge">Complete Analysis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="result-grid">', unsafe_allow_html=True)
    if reading.palm_photo_jpeg:
        st.markdown('<div class="media-card">', unsafe_allow_html=True)
        st.image(reading.palm_photo_jpeg, caption="📷 Your Original Palm Photo", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    if reading.contour_png:
        st.markdown('<div class="media-card">', unsafe_allow_html=True)
        st.image(reading.contour_png, caption="🖼️ AI-Generated Palm Print", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    overview_tab, lines_tab, insights_tab, artwork_tab, full_tab = st.tabs([
        "🌟 Overview", 
        "📏 Palm Lines", 
        "💡 Insights", 
        "🎨 Artwork", 
        "📖 Full Guide"
    ])

    with overview_tab:
        st.markdown("### Quick Summary")
        render_card("At a Glance", section_body(sections, "At a Glance"))
        render_card("Your Unique Path", section_body(sections, "Your Path"))

    with lines_tab:
        st.markdown("### Detailed Line Analysis")
        line_sections = matching_sections(sections, ("line", "heart", "head", "life", "fate", "sun"))
        if not line_sections:
            line_sections = {"Your Palm Lines": section_body(sections, "Your Palm Lines")}
        for title, body in line_sections.items():
            render_card(title, body)

    with insights_tab:
        st.markdown("### Personal Insights & Guidance")
        insight_sections = matching_sections(sections, ("strength", "challenge", "love", "career", "guidance", "means"))
        if not insight_sections:
            insight_sections = {"What This Means For You": section_body(sections, "What This Means For You")}
        for title, body in insight_sections.items():
            render_card(title, body)

    with artwork_tab:
        st.markdown("### Visual Elements")
        col1, col2 = st.columns(2)
        with col1:
            if reading.palm_photo_jpeg:
                st.image(reading.palm_photo_jpeg, caption="📷 Original Capture", use_container_width=True)
        with col2:
            if reading.contour_png:
                st.image(reading.contour_png, caption="🎨 AI Enhancement", use_container_width=True)
        
        render_card("Palm Print Analysis", section_body(sections, "Your Palm Print"))

    with full_tab:
        st.markdown("### Complete Reading Report")
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown(reading.report_markdown)
        st.markdown("</div>", unsafe_allow_html=True)

    # Enhanced download section
    st.markdown(
        """
        <div class="section-header">
            <h3 class="section-title">📥 Save Your Reading</h3>
            <div class="section-badge">Downloads</div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    pdf_bytes = create_pdf_bytes(reading)
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "📄 Download Complete PDF Report",
            data=pdf_bytes,
            file_name="palmora-reading-report.pdf",
            mime="application/pdf",
            use_container_width=True,
            help="Get your complete palm reading as a beautiful PDF document"
        )
    with col_b:
        if reading.contour_png:
            st.download_button(
                "🖼️ Download Palm Print Art",
                data=reading.contour_png,
                file_name="palmora-palm-print-artwork.png",
                mime="image/png",
                use_container_width=True,
                help="Save the AI-generated palm print visualization"
            )


def render_step_progress(current_step: int, total_steps: int = 4) -> None:
    """Render a visual progress indicator for the step wizard."""
    st.markdown(
        f"""
        <div class="step-progress">
            <div class="progress-header">
                <h3>Palm Reading Progress</h3>
                <span class="step-counter">Step {current_step} of {total_steps}</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {(current_step / total_steps) * 100}%"></div>
            </div>
            <div class="step-labels">
                <span class="{'active' if current_step >= 1 else ''}">Setup</span>
                <span class="{'active' if current_step >= 2 else ''}">Capture</span>
                <span class="{'active' if current_step >= 3 else ''}">Analyze</span>
                <span class="{'active' if current_step >= 4 else ''}">Results</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step_navigation(current_step: int, can_go_next: bool = False, can_go_back: bool = True) -> dict:
    """Render navigation controls and return user actions."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_step > 1 and can_go_back:
            go_back = st.button("← Previous Step", use_container_width=True)
        else:
            go_back = False
    
    with col3:
        if current_step < 4 and can_go_next:
            go_next = st.button("Next Step →", type="primary", use_container_width=True)
        else:
            go_next = False
    
    return {"next": go_next, "back": go_back}


def render_step_1_setup() -> dict:
    """Step 1: Introduction and Setup"""
    st.markdown(
        """
        <div class="step-card">
            <div class="step-header">
                <div class="step-icon">🚀</div>
                <div>
                    <h2>Welcome to Palmora</h2>
                    <p>Professional AI-powered palm reading</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Feature overview
    st.markdown("### What You'll Get")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🔍 AI Analysis**
        - Advanced palm line detection
        - Professional quality assessment
        - Detailed feature analysis
        """)
    
    with col2:
        st.markdown("""
        **📊 Comprehensive Report**
        - Personalized insights
        - Interactive visualizations
        - Downloadable PDF guide
        """)
    
    with col3:
        st.markdown("""
        **🎨 Visual Artwork**
        - AI-generated palm art
        - Enhanced line drawings
        - Professional presentation
        """)
    
    # Settings and configuration
    st.markdown("---")
    st.markdown("### Configuration")
    
    with st.expander("⚙️ Advanced Settings", expanded=False):
        vision_model = st.text_input(
            "Vision Model", 
            value=os.getenv("PALM_READER_VISION_MODEL", DEFAULT_VISION_MODEL),
            help="AI model for palm image analysis"
        )
        image_model = st.text_input(
            "Image Model", 
            value=os.getenv("PALM_READER_IMAGE_MODEL", DEFAULT_IMAGE_MODEL),
            help="AI model for palm print generation"
        )
    
    # Privacy and consent
    st.markdown("### Privacy & Terms")
    
    privacy_info = st.expander("🛡️ Privacy Information", expanded=False)
    with privacy_info:
        st.markdown("""
        - **Data Processing:** Images are processed temporarily for analysis only
        - **Storage:** No permanent storage of palm images
        - **AI Provider:** Images sent to OpenAI for processing
        - **Purpose:** Entertainment and insight only, not for medical advice
        - **Security:** All data handled according to privacy best practices
        """)
    
    consent = st.checkbox(
        "✅ I understand this is for entertainment purposes and consent to AI processing",
        value=False,
        key="step1_consent"
    )
    
    # Check if API key is available
    client = get_client()
    api_available = client is not None
    
    if not api_available:
        st.error("🔑 **API Key Required:** Please configure your OpenAI API key to continue.")
    
    return {
        "consent": consent,
        "api_available": api_available,
        "vision_model": vision_model,
        "image_model": image_model,
        "can_continue": consent and api_available
    }


def render_step_2_capture() -> dict:
    """Step 2: Image Capture and Upload"""
    st.markdown(
        """
        <div class="step-card">
            <div class="step-header">
                <div class="step-icon">📷</div>
                <div>
                    <h2>Capture Your Palm</h2>
                    <p>Take or upload a clear photo of your palm</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Photo guide
    render_photo_guide()
    
    # Capture interface with tabs
    capture_method = st.radio(
        "Choose capture method:",
        ["📷 Live Camera", "📁 Upload Image"],
        horizontal=True,
        key="capture_method"
    )
    
    camera_photo = None
    uploaded = None
    
    if capture_method == "📷 Live Camera":
        st.markdown("### 📸 Live Camera Capture")
        st.info("📱 **Mobile Tip:** Hold your device steady and use good lighting for best results!")
        camera_photo = st.camera_input(
            "📷 Capture Your Palm", 
            help="Position your open palm facing the camera"
        )
    
    else:  # Upload Image
        st.markdown("### 📁 Upload Image")
        st.info("📱 **Mobile Tip:** Tap to select from your photo gallery or take a new photo!")
        uploaded = st.file_uploader(
            "📎 Select your palm image",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"],
            help="Supported: PNG, JPG, JPEG, WebP, BMP, TIFF",
            accept_multiple_files=False,
            key="palm_uploader"
        )
        
        if uploaded:
            # Validate and enhance uploaded image
            is_valid, message, metrics = validate_uploaded_image(uploaded)
            
            if is_valid and metrics:
                st.success("✅ Image uploaded successfully!")
                
                # Quality metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Resolution", f"{metrics['width']}×{metrics['height']}")
                with col2:
                    st.metric("Format", metrics['format'])
                with col3:
                    st.metric("Size", f"{metrics['size']/1024:.1f} KB")
                with col4:
                    st.metric("Quality", f"{metrics['quality_score']}%")
                
                # Enhancement options
                if st.checkbox("🔧 Apply AI Enhancement", value=True, key="enhance_check"):
                    with st.spinner("Enhancing image..."):
                        enhanced_bytes = enhance_image(uploaded.getvalue())
                        st.session_state['enhanced_image'] = enhanced_bytes
                        st.success("✅ Image enhanced!")
            else:
                st.error(f"❌ {message}")
                uploaded = None
    
    source = camera_photo or uploaded
    image_ready = source is not None
    
    # Preview selected image
    if source:
        if 'enhanced_image' in st.session_state and uploaded:
            preview_bytes = st.session_state['enhanced_image']
            st.success("🔧 Showing enhanced version")
        else:
            preview_bytes = source.getvalue()
        
        st.markdown("### Selected Image Preview")
        st.image(preview_bytes, caption="Your palm photo", use_container_width=True)
    
    return {
        "source": source,
        "image_ready": image_ready,
        "camera_photo": camera_photo,
        "uploaded": uploaded
    }


def render_step_3_analyze(source, vision_model: str, image_model: str) -> dict:
    """Step 3: AI Analysis Configuration and Processing"""
    st.markdown(
        """
        <div class="step-card">
            <div class="step-header">
                <div class="step-icon">🤖</div>
                <div>
                    <h2>AI Analysis</h2>
                    <p>Configure and run palm analysis</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Check if analysis is complete
    if st.session_state.get("analysis_complete", False):
        st.success("🎉 **Analysis Complete!** Your palm reading is ready!")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("✨ View My Results", type="primary", use_container_width=True):
                st.session_state.current_step = 4
                st.rerun()
        with col2:
            if st.button("🔄 Analyze Again", use_container_width=True):
                # Reset analysis state
                st.session_state.analysis_complete = False
                st.session_state.analysis_running = False
                if "reading" in st.session_state:
                    del st.session_state["reading"]
                st.rerun()
        
        return {"analysis_complete": True, "reading": st.session_state.get("reading")}
    
    # Check if analysis is running
    analysis_running = st.session_state.get("analysis_running", False)
    
    if not analysis_running:
        # Analysis configuration
        st.markdown("### Analysis Options")
        
        col1, col2 = st.columns(2)
        with col1:
            detailed_analysis = st.checkbox(
                "🔬 Detailed Analysis", 
                value=True, 
                help="Include comprehensive palm line analysis"
            )
        with col2:
            generate_artwork = st.checkbox(
                "🎨 Generate Artwork", 
                value=True, 
                help="Create artistic palm print visualization"
            )
        
        # Start analysis button
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
                # Set analysis as running and store options
                st.session_state.analysis_running = True
                st.session_state.detailed_analysis = detailed_analysis
                st.session_state.generate_artwork = generate_artwork
                st.rerun()
        
        return {"analysis_complete": False, "reading": None}
    
    else:
        # Analysis is running - show progress and run analysis
        st.warning("🔄 **Analysis in Progress** - Please wait...")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.button("🚀 Analysis Running...", disabled=True, use_container_width=True)
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.analysis_running = False
                st.rerun()
        
        # Run the actual analysis
        try:
            client = get_client()
            detailed_analysis = st.session_state.get("detailed_analysis", True)
            generate_artwork = st.session_state.get("generate_artwork", True)
            
            # Get image bytes
            if 'enhanced_image' in st.session_state:
                raw_image_bytes = st.session_state['enhanced_image']
            else:
                raw_image_bytes = source.getvalue()
            
            optimized_bytes, mime_type = optimize_upload_image(raw_image_bytes)
            
            with st.status("🔮 AI Analysis in Progress...", expanded=True) as status:
                # Validation
                status.write("🔍 Step 1/4: Validating image quality...")
                validation = validate_palm_photo(client, optimized_bytes, mime_type, model=vision_model)
                
                if not validation.is_valid:
                    status.update(label="❌ Analysis failed", state="error")
                    st.session_state.analysis_running = False
                    
                    st.error("🚫 Image quality insufficient for analysis")
                    
                    if validation.issues:
                        for issue in validation.issues:
                            st.write(f"❌ {issue}")
                    
                    return {"analysis_complete": False, "reading": None}
                
                # Analysis
                status.write(f"✅ Validation passed ({validation.score}/100)")
                status.write("🧠 Step 2/4: Analyzing palm features...")
                
                report = generate_report(client, optimized_bytes, mime_type, model=vision_model)
                status.write("📊 Palm analysis complete!")
                
                # Artwork generation
                contour = None
                if generate_artwork:
                    status.write("🎨 Step 3/4: Generating artwork...")
                    try:
                        contour = generate_contour_image(client, report, image_bytes=optimized_bytes, model=image_model)
                        status.write("🖼️ Artwork generated!")
                    except Exception as exc:
                        status.write(f"⚠️ Artwork generation failed: {exc}")
                
                # Finalization
                status.write("📋 Step 4/4: Finalizing report...")
                status.update(label="🎉 Analysis Complete!", state="complete")
            
            # Store results and update state
            reading_result = PalmReading(
                report_markdown=report,
                palm_photo_jpeg=optimized_bytes,
                contour_png=contour,
            )
            st.session_state["reading"] = reading_result
            st.session_state.analysis_running = False
            st.session_state.analysis_complete = True
            
            st.balloons()
            st.rerun()  # Refresh to show completion state
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.session_state.analysis_running = False
            return {"analysis_complete": False, "reading": None}
    
    return {"analysis_complete": False, "reading": None}


def render_step_4_results() -> dict:
    """Step 4: Display Beautiful Results with Enhanced Design"""
    st.markdown(
        """
        <div class="results-hero">
            <div class="results-header">
                <div class="results-icon">✨</div>
                <div class="results-content">
                    <h1>Your Personalized Palm Reading</h1>
                    <p>Comprehensive AI-powered analysis complete</p>
                </div>
                <div class="results-badge">🎉 Complete</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if "reading" in st.session_state:
        reading = st.session_state["reading"]
        
        # Beautiful results summary cards
        st.markdown("### 📊 Reading Summary")
        
        # Quick stats in beautiful cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                """
                <div class="summary-card">
                    <div class="card-icon">🔍</div>
                    <h3>Analysis Complete</h3>
                    <p>Professional AI palm reading with detailed insights and recommendations</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with col2:
            st.markdown(
                """
                <div class="summary-card">
                    <div class="card-icon">🎨</div>
                    <h3>Visual Artwork</h3>
                    <p>AI-generated palm print visualization with enhanced line detection</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with col3:
            st.markdown(
                """
                <div class="summary-card">
                    <div class="card-icon">📄</div>
                    <h3>PDF Report</h3>
                    <p>Downloadable comprehensive report with all insights and artwork</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        # Main results display
        render_report(reading)
        
        # Enhanced action section
        st.markdown(
            """
            <div class="action-section">
                <h2>🚀 What's Next?</h2>
                <p>Choose your next action or explore more features</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Action buttons in beautiful cards
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            st.markdown(
                """
                <div class="action-card">
                    <div class="action-icon">🔄</div>
                    <h3>New Reading</h3>
                    <p>Start a fresh palm reading session</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("🔄 Start New Reading", use_container_width=True, type="primary"):
                # Clear session and restart
                for key in list(st.session_state.keys()):
                    if key.startswith(('reading', 'enhanced_image', 'current_step', 'analysis')):
                        del st.session_state[key]
                st.session_state.current_step = 1
                st.rerun()
        
        with action_col2:
            st.markdown(
                """
                <div class="action-card">
                    <div class="action-icon">📤</div>
                    <h3>Share Results</h3>
                    <p>Share your reading with friends</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("📤 Share Reading", use_container_width=True):
                st.info("🚀 **Coming Soon!** Social sharing features will be available in the next update.")
        
        with action_col3:
            st.markdown(
                """
                <div class="action-card">
                    <div class="action-icon">💾</div>
                    <h3>Save & Export</h3>
                    <p>Download your complete report</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Enhanced PDF download
            pdf_bytes = create_pdf_bytes(reading)
            st.download_button(
                "📄 Download Premium PDF",
                data=pdf_bytes,
                file_name=f"palmora-reading-{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Download your complete palm reading as a beautifully formatted PDF"
            )
        
        # Additional features section
        st.markdown("---")
        st.markdown("### 🌟 Explore More Features")
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            with st.expander("📈 Reading History", expanded=False):
                st.info("""
                **Coming Soon: Personal Dashboard**
                • Track multiple readings over time
                • Compare palm changes and insights
                • Export historical data
                • Personal growth tracking
                """)
        
        with feature_col2:
            with st.expander("🔬 Advanced Analysis", expanded=False):
                st.info("""
                **Premium Features Available:**
                • Detailed compatibility readings
                • Career and relationship insights
                • Health and wellness indicators
                • Personalized recommendations
                """)
        
        # Feedback section
        st.markdown("### 💬 How was your experience?")
        
        feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
        
        with feedback_col1:
            if st.button("😍 Amazing!", use_container_width=True):
                st.success("🙏 Thank you! We're glad you loved your reading!")
        
        with feedback_col2:
            if st.button("👍 Good", use_container_width=True):
                st.success("✨ Thanks for the feedback! We're always improving.")
        
        with feedback_col3:
            if st.button("📝 Suggest Improvements", use_container_width=True):
                st.info("💡 Your suggestions help us improve! Contact us at feedback@palmora.ai")
    
    else:
        st.error("❌ **No reading data found.** Please go back and complete the analysis.")
        if st.button("← Return to Analysis", type="primary"):
            st.session_state.current_step = 3
            st.rerun()
    
    return {"completed": True}


def main() -> None:
    # Initialize session state for step tracking
    if "current_step" not in st.session_state:
        st.session_state.current_step = 1
    
    # Render header
    render_header()
    
    # Render step progress
    render_step_progress(st.session_state.current_step)
    
    # Step-based navigation
    current_step = st.session_state.current_step
    
    if current_step == 1:
        # Step 1: Setup and Introduction
        step_result = render_step_1_setup()
        navigation = render_step_navigation(1, can_go_next=step_result["can_continue"], can_go_back=False)
        
        if navigation["next"]:
            st.session_state.current_step = 2
            st.session_state.step1_data = step_result
            st.rerun()
    
    elif current_step == 2:
        # Step 2: Image Capture
        step_result = render_step_2_capture()
        navigation = render_step_navigation(2, can_go_next=step_result["image_ready"], can_go_back=True)
        
        if navigation["next"]:
            st.session_state.current_step = 3
            st.session_state.step2_data = step_result
            st.rerun()
        elif navigation["back"]:
            st.session_state.current_step = 1
            st.rerun()
    
    elif current_step == 3:
        # Step 3: Analysis
        if "step2_data" in st.session_state and st.session_state.step2_data["image_ready"]:
            step1_data = st.session_state.get("step1_data", {})
            step2_data = st.session_state.step2_data
            
            step_result = render_step_3_analyze(
                step2_data["source"],
                step1_data.get("vision_model", DEFAULT_VISION_MODEL),
                step1_data.get("image_model", DEFAULT_IMAGE_MODEL)
            )
            
            # Check if analysis is complete or running
            analysis_complete = st.session_state.get("analysis_complete", False)
            analysis_running = st.session_state.get("analysis_running", False)
            
            # Only show navigation if not running analysis
            if not analysis_running:
                navigation = render_step_navigation(3, can_go_next=analysis_complete, can_go_back=True)
                
                if navigation["next"] and analysis_complete:
                    st.session_state.current_step = 4
                    st.session_state.step3_data = step_result
                    st.rerun()
                elif navigation["back"]:
                    # Clear analysis state when going back
                    st.session_state.analysis_complete = False
                    st.session_state.analysis_running = False
                    st.session_state.current_step = 2
                    st.rerun()
        else:
            st.error("No image data found. Please go back to step 2.")
            if st.button("← Go Back to Step 2"):
                st.session_state.current_step = 2
                st.rerun()
    
    elif current_step == 4:
        # Step 4: Results
        step_result = render_step_4_results()
        # No navigation needed - user can restart from here
    
    # Sidebar with progress and info
    with st.sidebar:
        st.markdown("### 📊 Session Progress")
        
        # Step indicators - mobile-friendly
        steps = [
            ("🚀", "Setup", 1),
            ("📷", "Capture", 2), 
            ("🤖", "Analyze", 3),
            ("✨", "Results", 4)
        ]
        
        for icon, name, step_num in steps:
            if step_num == current_step:
                st.markdown(f"**{icon} {name}** ← Now")
            elif step_num < current_step:
                st.markdown(f"✅ {icon} {name}")
            else:
                st.markdown(f"⚪ {icon} {name}")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        **Palmora v2026.1**  
        Professional AI palm reading with step-by-step guidance.
        
        Each step is designed to ensure the best possible analysis results.
        """)
        
        if st.button("🔄 Restart Session", use_container_width=True):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
