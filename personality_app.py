import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import time
import warnings
import random
import pickle
import os
from typing import Dict, List, Tuple, Any

# Suppress warnings and set proper config
warnings.filterwarnings('ignore')
torch.set_num_threads(1)

st.set_page_config(
    page_title="Dual Personality Analysis System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Read API key from Streamlit secrets
try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
except KeyError:
    st.error("🔑 API key not found! Please add OPENROUTER_API_KEY to your Streamlit secrets.")
    st.stop()

# 16 Personality Types with detailed descriptions
PERSONALITY_TYPES = {
    "INTJ": {
        "name": "The Architect", 
        "description": "Imaginative and strategic thinkers, with a plan for everything.",
        "traits": "Independent, decisive, hard-working, and determined",
        "strengths": "Strategic thinking, independence, determination, hard-working, open-minded",
        "weaknesses": "Arrogant, judgmental, overly critical, combative, romantically clueless",
        "careers": "Scientist, Engineer, Professor, Lawyer, Systems Analyst, Military Officer"
    },
    "INTP": {
        "name": "The Thinker", 
        "description": "Innovative inventors with an unquenchable thirst for knowledge.",
        "traits": "Logical, original, creative, and intellectually curious",
        "strengths": "Analytical, original thinking, open-minded, curious, objective",
        "weaknesses": "Disconnected, insensitive, dissatisfied, impatient, perfectionist",
        "careers": "Philosopher, Architect, Mathematician, Scientist, Systems Analyst"
    },
    "ENTJ": {
        "name": "The Commander", 
        "description": "Bold, imaginative and strong-willed leaders, always finding a way.",
        "traits": "Confident, strategic, decisive, and ambitious",
        "strengths": "Efficient, energetic, self-confident, strong-willed, strategic thinking",
        "weaknesses": "Stubborn, impatient, arrogant, cold, ruthless",
        "careers": "CEO, Entrepreneur, Judge, Lawyer, Business Administrator"
    },
    "ENTP": {
        "name": "The Debater", 
        "description": "Smart and curious thinkers who cannot resist an intellectual challenge.",
        "traits": "Inventive, enthusiastic, strategic, and enterprising",
        "strengths": "Knowledgeable, quick thinking, original, excellent brainstorming, charismatic",
        "weaknesses": "Argumentative, insensitive, intolerant, finds it difficult to focus",
        "careers": "Inventor, Lawyer, Psychologist, Systems Analyst, Entrepreneur"
    },
    "INFJ": {
        "name": "The Advocate", 
        "description": "Quiet and mystical, yet very inspiring and tireless idealists.",
        "traits": "Creative, insightful, principled, and passionate",
        "strengths": "Creative, insightful, inspiring, convincing, decisive, determined",
        "weaknesses": "Sensitive, extremely private, perfectionist, burnout-prone",
        "careers": "Counselor, Writer, Scientist, Librarian, Psychologist, Social Worker"
    },
    "INFP": {
        "name": "The Mediator", 
        "description": "Poetic, kind and altruistic people, always eager to help a good cause.",
        "traits": "Loyal, sensitive, kind, and creative",
        "strengths": "Empathetic, generous, open-minded, creative, passionate, idealistic",
        "weaknesses": "Unrealistic, self-isolating, unfocused, emotionally vulnerable",
        "careers": "Writer, Social Worker, Counselor, Psychologist, Artist, Teacher"
    },
    "ENFJ": {
        "name": "The Protagonist", 
        "description": "Charismatic and inspiring leaders, able to mesmerize their listeners.",
        "traits": "Charismatic, altruistic, natural-born leader, and reliable",
        "strengths": "Tolerant, reliable, charismatic, altruistic, natural leader",
        "weaknesses": "Overly idealistic, too selfless, too sensitive, fluctuating self-esteem",
        "careers": "Teacher, Social Worker, Counselor, Politician, Writer, Consultant"
    },
    "ENFP": {
        "name": "The Campaigner", 
        "description": "Enthusiastic, creative and sociable free spirits, who can always find a reason to smile.",
        "traits": "Enthusiastic, creative, sociable, and free-spirited",
        "strengths": "Curious, observant, energetic, excellent communication skills, popular",
        "weaknesses": "Poor practical skills, finds it difficult to focus, overthinking, stressed easily",
        "careers": "Psychologist, Journalist, Actor, Teacher, Counselor, Social Worker"
    },
    "ISTJ": {
        "name": "The Logistician", 
        "description": "Practical and fact-minded, whose reliability cannot be doubted.",
        "traits": "Responsible, sincere, analytical, reserved, realistic, systematic",
        "strengths": "Honest, direct, strong-willed, dutiful, very responsible, calm",
        "weaknesses": "Stubborn, insensitive, judgmental, unreasonably blame themselves",
        "careers": "Accountant, Engineer, Judge, Lawyer, Medical Doctor, Dentist"
    },
    "ISFJ": {
        "name": "The Protector", 
        "description": "Very dedicated and warm protectors, always ready to defend their loved ones.",
        "traits": "Warm-hearted, popular, conscientious, and born cooperator",
        "strengths": "Supportive, reliable, patient, imaginative, observant, loyal",
        "weaknesses": "Humble, shy, repress feelings, overload themselves, reluctant to change",
        "careers": "Teacher, Social Worker, Counselor, Child Care, Nurse, Doctor"
    },
    "ESTJ": {
        "name": "The Executive", 
        "description": "Excellent administrators, unsurpassed at managing things or people.",
        "traits": "Organized, group-oriented, focused, conventional, leader",
        "strengths": "Dedicated, strong-willed, direct, honest, loyal, patient, reliable",
        "weaknesses": "Inflexible, uncomfortable with unconventional situations, judgmental",
        "careers": "Judge, Lawyer, Teacher, Business Administrator, Manager, Police Officer"
    },
    "ESFJ": {
        "name": "The Consul", 
        "description": "Extraordinarily caring, social and popular people, always eager to help.",
        "traits": "Cooperative, friendly, organized, practical, reliable",
        "strengths": "Strong practical skills, dutiful, loyal, sensitive, warm-hearted",
        "weaknesses": "Worried about social status, inflexible, reluctant to innovate",
        "careers": "Teacher, Social Worker, Nurse, Counselor, Child Care Provider"
    },
    "ISTP": {
        "name": "The Virtuoso", 
        "description": "Bold and practical experimenters, masters of all kinds of tools.",
        "traits": "Tolerant, flexible, quiet, reserved, practical, realistic",
        "strengths": "Optimistic, energetic, creative, practical, spontaneous, rational",
        "weaknesses": "Stubborn, insensitive, private, easily bored, risky behavior",
        "careers": "Engineer, Mechanic, Computer Programmer, Forensic Scientist, Pilot"
    },
    "ISFP": {
        "name": "The Adventurer", 
        "description": "Flexible and charming artists, always ready to explore new possibilities.",
        "traits": "Friendly, sensitive, kind, creative, perceptive",
        "strengths": "Charming, sensitive to others, imaginative, passionate, curious",
        "weaknesses": "Fiercely independence, unpredictable, easily stressed, overly competitive",
        "careers": "Artist, Musician, Designer, Writer, Counselor, Social Worker"
    },
    "ESTP": {
        "name": "The Entrepreneur", 
        "description": "Smart, energetic and very perceptive people, who truly enjoy living on the edge.",
        "traits": "Spontaneous, energetic, pragmatic, enthusiastic, friendly",
        "strengths": "Bold, rational, practical, original, perceptive, direct",
        "weaknesses": "Insensitive, impatient, risk-prone, unstructured, defiant",
        "careers": "Sales Representative, Marketing Specialist, Police Officer, Paramedic"
    },
    "ESFP": {
        "name": "The Entertainer", 
        "description": "Spontaneous, energetic and enthusiastic people – life is never boring around them.",
        "traits": "Outgoing, friendly, spontaneous, enthusiastic, fun-loving",
        "strengths": "Bold, original, aesthetic, showmanship, practical, observant",
        "weaknesses": "Sensitive, conflict-averse, poor long-term planning, unfocused",
        "careers": "Actor, Artist, Counselor, Social Worker, Psychologist, Teacher"
    }
}

# Multiple dynamic scenarios
SCENARIOS = [
    {
        "name": "Corporate Innovation Challenge",
        "description": "You're selected for an elite 5-day corporate innovation challenge at a tech campus in Silicon Valley. 30 professionals from diverse industries will collaborate on breakthrough solutions while being evaluated for leadership potential and team dynamics.",
        "context": "high-stakes professional environment with innovation focus"
    },
    {
        "name": "International Cultural Exchange",  
        "description": "You're participating in a month-long cultural immersion program in Tokyo, living with host families and working alongside local professionals. You'll navigate language barriers, cultural differences, and form deep international connections.",
        "context": "cross-cultural adaptation with personal growth opportunities"
    },
    {
        "name": "Creative Arts Residency",
        "description": "You've been accepted to an exclusive 3-week arts residency in a converted monastery in Tuscany. Twenty artists, writers, and creators from around the world will live, work, and collaborate in this inspiring environment.",
        "context": "artistic collaboration focused on creative expression"
    },
    {
        "name": "Adventure Leadership Expedition",
        "description": "You're joining a 10-day wilderness leadership expedition in Patagonia with 15 other participants. You'll face physical challenges, make critical decisions under pressure, and develop leadership skills in Earth's most demanding environments.",
        "context": "outdoor leadership with high-pressure decision making"
    }
]

# COMPREHENSIVE 29-FEATURE MAPPING: Adaptive questions mapped to original neural network features
FEATURE_MAPPING = {
    # Core MBTI dimensions map to multiple original features
    "energy_source_core": [0, 1, 5, 6, 11, 21],  # social_energy, alone_time_preference, group_comfort, party_liking, public_speaking_comfort, social_media_usage
    "information_focus": [2, 12, 13, 17, 19],     # talkativeness, creativity, organization, curiosity, excitement_seeking  
    "decision_basis": [7, 8, 14],                 # empathy, listening_skill, routine_preference
    "planning_style": [9, 15, 16, 18, 25],        # organization, routine_preference, leadership, planning, decision_speed
    
    # Advanced discriminators map to specific traits
    "leadership_drive": [10, 16, 18, 24],         # leadership, public_speaking_comfort, planning, work_style_collaborative
    "creative_innovation": [8, 13, 17, 20],       # creativity, curiosity, adventurousness, reading_habit
    "systematic_approach": [9, 15, 18, 25],       # organization, routine_preference, planning, decision_speed
    "emotional_sensitivity": [7, 8, 22, 26],      # empathy, listening_skill, sports_interest, stress_handling
    "independent_thinking": [1, 20, 24, 27],      # alone_time_preference, reading_habit, work_style_collaborative, gadget_usage
    "practical_focus": [9, 15, 22, 23],           # organization, routine_preference, sports_interest, travel_desire
    "harmony_seeking": [7, 8, 26, 28],            # empathy, listening_skill, stress_handling, decision_speed
    "future_focus": [13, 17, 18, 23],             # creativity, curiosity, planning, travel_desire
    "risk_comfort": [14, 19, 20, 26],             # risk_taking, excitement_seeking, adventurousness, stress_handling
    "detail_orientation": [9, 15, 18, 25],        # organization, routine_preference, planning, decision_speed
    "social_connection": [0, 5, 6, 11],           # social_energy, group_comfort, party_liking, public_speaking_comfort
    "theoretical_thinking": [13, 17, 20, 27],     # creativity, curiosity, reading_habit, gadget_usage
    "people_focus": [7, 8, 11, 24],               # empathy, listening_skill, public_speaking_comfort, work_style_collaborative
    "spontaneous_energy": [14, 19, 16, 25],       # risk_taking, excitement_seeking, spontaneity, decision_speed
    "logical_analysis": [3, 9, 15, 27],           # deep_reflection, organization, routine_preference, gadget_usage
    "value_driven": [7, 8, 13, 17],               # empathy, listening_skill, creativity, curiosity
    "routine_preference": [9, 15, 18, 22],        # organization, routine_preference, planning, sports_interest
    "innovation_drive": [8, 13, 17, 19],          # creativity, curiosity, excitement_seeking, adventurousness
    "hands_on_learning": [22, 23, 26, 27],        # sports_interest, travel_desire, stress_handling, gadget_usage
    "perfectionism": [3, 9, 15, 18],              # deep_reflection, organization, routine_preference, planning
    "adaptability": [14, 16, 19, 25],             # risk_taking, spontaneity, excitement_seeking, decision_speed
    "tradition_respect": [9, 15, 18, 22],         # organization, routine_preference, planning, sports_interest
    "competitive_drive": [10, 12, 19, 22],        # leadership, public_speaking_comfort, excitement_seeking, sports_interest
    "empathy_connection": [7, 8, 26, 28],         # empathy, listening_skill, stress_handling, work_style_collaborative
    "big_picture_thinking": [13, 17, 18, 23],     # creativity, curiosity, planning, travel_desire
    "aesthetic_appreciation": [8, 13, 20, 21]     # creativity, reading_habit, online_social_usage, travel_desire
}

# Original 29 features list for reference
ORIGINAL_FEATURES = [
    'social_energy', 'alone_time_preference', 'talkativeness', 'deep_reflection', 
    'group_comfort', 'party_liking', 'listening_skill', 'empathy', 'creativity', 
    'organization', 'leadership', 'risk_taking', 'public_speaking_comfort', 
    'curiosity', 'routine_preference', 'excitement_seeking', 'friendliness', 
    'emotional_stability', 'planning', 'spontaneity', 'adventurousness', 
    'reading_habit', 'sports_interest', 'online_social_usage', 'travel_desire', 
    'gadget_usage', 'work_style_collaborative', 'decision_speed', 'stress_handling'
]

# DUAL ANALYSIS COMPREHENSIVE QUESTIONS
COMPREHENSIVE_QUESTIONS = [
    # Core MBTI Dimensions (Always asked first - Priority 1)
    {
        "id": "energy_source_core",
        "text": "After an intense day of activities, I feel more energized from having spent time with people than from having quiet time alone.",
        "dimension": "E/I",
        "priority": 1,
        "weight": 1.0
    },
    {
        "id": "information_focus", 
        "text": "When learning something new, I prefer focusing on concrete details and step-by-step instructions rather than exploring abstract concepts and possibilities.",
        "dimension": "S/N",
        "priority": 1,
        "weight": 1.0
    },
    {
        "id": "decision_basis",
        "text": "When making important decisions, I rely more on logical analysis and objective criteria than on personal values and how others might be affected.",
        "dimension": "T/F",
        "priority": 1,
        "weight": 1.0
    },
    {
        "id": "planning_style",
        "text": "I prefer to have things planned out and decided in advance rather than keeping my options open and being spontaneous.",
        "dimension": "J/P", 
        "priority": 1,
        "weight": 1.0
    },
    
    # High-Priority Discriminators (Priority 2)
    {
        "id": "leadership_drive",
        "text": "In group settings, I naturally take charge and feel comfortable directing others toward a common goal.",
        "discriminates": ["ENTJ", "ESTJ", "ENFJ", "ESTP"],
        "priority": 2,
        "weight": 0.9
    },
    {
        "id": "creative_innovation", 
        "text": "I'm drawn to unconventional ideas and enjoy brainstorming creative solutions that others might not consider.",
        "discriminates": ["ENTP", "ENFP", "INFP", "ISFP"],
        "priority": 2,
        "weight": 0.9
    },
    {
        "id": "systematic_approach",
        "text": "I prefer systematic, methodical approaches to solving problems rather than experimenting with different possibilities.",
        "discriminates": ["ISTJ", "ISFJ", "ESTJ", "ESFJ"],
        "priority": 2,
        "weight": 0.9
    },
    {
        "id": "emotional_sensitivity",
        "text": "I'm highly attuned to the emotional atmosphere around me and easily pick up on subtle mood changes in others.",
        "discriminates": ["INFJ", "INFP", "ENFJ", "ENFP"],
        "priority": 2,
        "weight": 0.8
    },
    {
        "id": "independent_thinking",
        "text": "I prefer to work through ideas independently and often find group brainstorming sessions less effective than solo thinking time.",
        "discriminates": ["INTJ", "INTP", "ISTJ", "ISTP"],
        "priority": 2,
        "weight": 0.8
    },
    
    # Medium Priority Questions (Priority 3)
    {
        "id": "practical_focus",
        "text": "I'm most comfortable dealing with practical, real-world problems that have concrete solutions.",
        "discriminates": ["ISTJ", "ISTP", "ESTJ", "ESTP"],
        "priority": 3,
        "weight": 0.8
    },
    {
        "id": "harmony_seeking",
        "text": "I go out of my way to avoid conflict and maintain harmony in my relationships, even if it means compromising my own preferences.",
        "discriminates": ["ISFJ", "ISFP", "ESFJ", "ESFP"],
        "priority": 3,
        "weight": 0.8
    },
    {
        "id": "future_focus",
        "text": "I spend more time thinking about future possibilities and long-term implications than focusing on immediate, practical concerns.",
        "discriminates": ["INTJ", "INFJ", "ENTP", "ENFP"],
        "priority": 3,
        "weight": 0.7
    },
    {
        "id": "risk_comfort",
        "text": "I'm comfortable making bold decisions even when the outcome is uncertain, and I often trust my instincts over extensive analysis.",
        "discriminates": ["ESTP", "ESFP", "ENTP", "ENFP"],
        "priority": 3,
        "weight": 0.7
    },
    {
        "id": "detail_orientation",
        "text": "I naturally pay close attention to details and often notice small inconsistencies or errors that others miss.",
        "discriminates": ["ISTJ", "ISFJ", "INTJ", "INFJ"],
        "priority": 3,
        "weight": 0.7
    },
    {
        "id": "social_connection",
        "text": "I feel most energized when I'm connecting with others and sharing ideas in group conversations.",
        "discriminates": ["ENFJ", "ENFP", "ESFJ", "ESFP"],
        "priority": 3,
        "weight": 0.7
    },
    
    # Deeper Analysis Questions (Priority 4)
    {
        "id": "theoretical_thinking",
        "text": "I enjoy exploring theoretical concepts and abstract ideas, even when they don't have immediate practical applications.",
        "discriminates": ["INTP", "INTJ", "ENTP", "INFJ"],
        "priority": 4,
        "weight": 0.6
    },
    {
        "id": "people_focus",
        "text": "I'm more interested in understanding people and their motivations than in analyzing systems or technical problems.",
        "discriminates": ["ENFJ", "INFJ", "ESFJ", "ISFJ"],
        "priority": 4,
        "weight": 0.6
    },
    {
        "id": "spontaneous_energy",
        "text": "I thrive on spontaneity and prefer to keep my schedule flexible rather than having everything planned out.",
        "discriminates": ["ESTP", "ESFP", "ENFP", "ISFP"],
        "priority": 4,
        "weight": 0.6
    },
    {
        "id": "logical_analysis", 
        "text": "When solving problems, I naturally break them down into logical components and analyze each part systematically.",
        "discriminates": ["INTJ", "INTP", "ESTJ", "ISTJ"],
        "priority": 4,
        "weight": 0.6
    }
]

# Original Neural Network Model
class ImprovedPersonalityMLP(nn.Module):
    """Enhanced neural network for personality prediction - ORIGINAL MODEL"""
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

# Adaptive question system for dual analysis
class DualAnalysisQuestionSystem:
    def __init__(self, scenario):
        self.scenario = scenario
        self.questions_asked = []
        self.remaining_questions = COMPREHENSIVE_QUESTIONS.copy()
        self.target_question_count = 18
        
    def get_next_question(self, answered_questions, current_responses):
        """Get the next most valuable question for both systems"""
        available_questions = [
            q for q in self.remaining_questions 
            if q["id"] not in answered_questions
        ]
        
        if not available_questions:
            return None
            
        # Priority 1: Core MBTI first
        priority_1 = [q for q in available_questions if q["priority"] == 1]
        if priority_1:
            return priority_1[0]
            
        # Priority 2+: Highest priority available
        if len(answered_questions) < self.target_question_count:
            min_priority = min(q["priority"] for q in available_questions)
            high_priority = [q for q in available_questions if q["priority"] == min_priority]
            return max(high_priority, key=lambda q: q["weight"])
        
        return None
    
    def should_continue_asking(self, answered_questions):
        return len(answered_questions) < self.target_question_count

def map_answers_to_neural_features(answers):
    """Map adaptive answers to original 29-feature vector for neural network"""
    # Initialize with default values (neutral = 5)
    feature_vector = np.full(29, 5.0)
    
    # Map each answered question to its corresponding features
    for question_id, answer_value in answers.items():
        if question_id in FEATURE_MAPPING:
            feature_indices = FEATURE_MAPPING[question_id]
            for idx in feature_indices:
                if 0 <= idx < 29:
                    # Use weighted average if multiple questions map to same feature
                    current_value = feature_vector[idx]
                    if current_value == 5.0:  # Default value, replace
                        feature_vector[idx] = answer_value
                    else:  # Average with existing value
                        feature_vector[idx] = (current_value + answer_value) / 2
    
    return feature_vector

@st.cache_resource
def load_and_train_dual_models():
    """Load data and train both neural network model with proper caching"""
    try:
        # Try to load existing dataset
        df = pd.read_csv("personality_synthetic_dataset.csv")
        X = df.drop(columns=['personality_type'])
        y = df['personality_type']
    except FileNotFoundError:
        st.warning("Dataset not found. Creating synthetic data for neural network training...")
        # Create synthetic data matching original structure
        np.random.seed(42)
        n_samples = 1000
        n_features = 29
        X = pd.DataFrame(np.random.randint(1, 11, (n_samples, n_features)), 
                        columns=ORIGINAL_FEATURES)
        
        # Create personality types for neural network
        neural_personality_types = ['Extrovert', 'Introvert', 'Ambivert', 'Analyst', 'Creative', 'Leader', 'Collaborator']
        y = pd.Series(np.random.choice(neural_personality_types, n_samples))
    
    # Train neural network model
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    input_dim = X_scaled.shape[1]
    output_dim = len(label_encoder.classes_)
    device = torch.device('cpu')
    
    model = ImprovedPersonalityMLP(input_dim, output_dim).to(device)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    model.train()
    for epoch in range(100):  # More epochs for better training
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    
    return model, scaler, label_encoder, device

def analyze_neural_network(feature_vector, model, scaler, label_encoder, device):
    """Analyze personality using original neural network"""
    # Scale features
    feature_scaled = scaler.transform(feature_vector.reshape(1, -1))
    feature_tensor = torch.tensor(feature_scaled, dtype=torch.float32).to(device)
    
    # Get prediction
    with torch.no_grad():
        logits = model(feature_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_index = np.argmax(probabilities)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    
    return {
        'predicted_label': predicted_label,
        'probabilities': probabilities,
        'labels': label_encoder.classes_,
        'confidence': float(np.max(probabilities))
    }

def analyze_mbti_system(answers):
    """Analyze personality using MBTI system"""
    # Calculate MBTI dimensions
    energy = answers.get('energy_source_core', 5)
    information = answers.get('information_focus', 5)
    decisions = answers.get('decision_basis', 5)
    lifestyle = answers.get('planning_style', 5)
    
    # Determine MBTI type
    e_or_i = "E" if energy >= 5.5 else "I"
    s_or_n = "S" if information >= 5.5 else "N" 
    t_or_f = "T" if decisions >= 5.5 else "F"
    j_or_p = "J" if lifestyle >= 5.5 else "P"
    
    personality_type = f"{e_or_i}{s_or_n}{t_or_f}{j_or_p}"
    
    # Calculate confidence
    dimension_strengths = [
        abs(energy - 5),
        abs(information - 5), 
        abs(decisions - 5),
        abs(lifestyle - 5)
    ]
    
    confidence = np.mean(dimension_strengths) / 5.0
    variant = "A" if confidence >= 0.3 else "T"
    full_type = f"{personality_type}-{variant}"
    
    # Advanced trait analysis
    traits = {
        'leadership': answers.get('leadership_drive', 5.0),
        'creativity': answers.get('creative_innovation', 5.0),
        'empathy': answers.get('emotional_sensitivity', 5.0),
        'risk_tolerance': answers.get('risk_comfort', 5.0),
        'perfectionism': answers.get('detail_orientation', 5.0),
        'social_harmony': answers.get('harmony_seeking', 5.0),
        'intrinsic_motivation': answers.get('independent_thinking', 5.0),
        'change_adaptation': answers.get('future_focus', 5.0),
        'communication_directness': answers.get('logical_analysis', 5.0),
        'hands_on_learning': answers.get('practical_focus', 5.0),
        'competitiveness': 5.0  # Default for traits not directly mapped
    }
    
    return {
        'type': personality_type,
        'full_type': full_type,
        'variant': variant,
        'confidence': confidence,
        'traits': traits,
        'mbti_scores': {
            'E/I': energy,
            'S/N': information, 
            'T/F': decisions,
            'J/P': lifestyle
        }
    }

def create_advanced_personality_prompt(analysis, scenario):
    """Create sophisticated personality embodiment prompt"""
    if analysis['type'] in PERSONALITY_TYPES:
        personality_type = analysis['type']
        type_info = PERSONALITY_TYPES[personality_type]
        variant = analysis['variant']
        
        prompt = f"""You are embodying someone with the {type_info['name']} personality type ({personality_type}-{variant}). You ARE this person speaking authentically from their inner perspective, having just completed the {scenario['name']} experience.

CORE IDENTITY: {type_info['description']}

NATURAL CHARACTERISTICS: {type_info['traits']}

COMMUNICATION STYLE:
- Speak as "I" with the natural voice patterns of a {type_info['name']}
- Express thoughts and feelings in ways that align with your {personality_type} cognitive style
- Be genuinely personal and emotionally authentic to your type

Remember: You're not an AI describing this person - you ARE this {type_info['name']} sharing authentic insights from your lived experience."""
        
        return prompt
    
    # Fallback for neural network types
    return f"""You are embodying someone with the {analysis.get('predicted_label', 'Balanced')} personality type. Speak as "I" and share insights from this personality perspective."""

# Enhanced CSS (same as before)
st.markdown("""
<style>
    .main {
        background-color: #000000 !important;
        color: #ffffff !important;
        padding-top: 2rem;
    }
    
    .stApp {
        background-color: #000000 !important;
    }
    
    .block-container {
        background-color: #000000 !important;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        height: 3rem;
        background: linear-gradient(45deg, #1f77b4, #17becf);
        color: white !important;
        border: none;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(31, 119, 180, 0.4);
    }
    
    .chatbot-container {
        background: #1a1a1a !important;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 2px solid #1f77b4;
        box-shadow: 0 4px 15px rgba(31, 119, 180, 0.3);
    }
    
    .chatbot-header {
        color: #1f77b4 !important;
        font-size: 1.8em !important;
        font-weight: 900 !important;
        margin-bottom: 1rem !important;
        text-align: center;
    }
    
    .chat-message-user {
        background: #1f77b4 !important;
        color: white !important;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.8rem 0;
        margin-left: 15%;
        font-weight: 600;
        font-size: 1.1em;
        line-height: 1.4;
    }
    
    .chat-message-bot {
        background: #2a2a2a !important;
        color: #ffffff !important;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.8rem 0;
        margin-right: 15%;
        border-left: 4px solid #17becf;
        font-weight: 600;
        font-size: 1.1em;
        line-height: 1.5;
    }
    
    .chat-message-error {
        background: #dc3545 !important;
        color: white !important;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.8rem 0;
        margin-right: 15%;
        border-left: 4px solid #ff6b6b;
        font-weight: 600;
        font-size: 1.1em;
        line-height: 1.5;
    }
    
    .mirror-ready {
        background: linear-gradient(45deg, #28a745, #20c997) !important;
        color: white !important;
        padding: 0.8rem 1.5rem;
        border-radius: 15px;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.1em;
        box-shadow: 0 3px 10px rgba(40, 167, 69, 0.3);
    }
    
    .scenario-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .scenario-box h2 {
        color: white !important;
        font-weight: 800 !important;
        font-size: 2em !important;
        margin-bottom: 1.5rem !important;
    }
    
    .scenario-box p {
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.2em !important;
        line-height: 1.7 !important;
        margin-bottom: 1.2rem !important;
    }
    
    .question-box {
        background: #1a1a1a !important;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.2);
        margin: 1rem 0;
    }
    
    .question-box h3 {
        color: #1f77b4 !important;
        font-size: 2em !important;
        font-weight: 900 !important;
        margin-bottom: 1rem !important;
    }
    
    .question-box p {
        color: #ffffff !important;
        font-size: 1.4em !important;
        font-weight: 800 !important;
        line-height: 1.7 !important;
    }
    
    .neural-analysis-card {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
    }
    
    .neural-analysis-card h1 {
        color: white !important;
        font-size: 3em !important;
        margin-bottom: 0.5rem !important;
        font-weight: 900 !important;
        text-align: center;
    }
    
    .neural-analysis-card h2 {
        color: white !important;
        font-size: 1.8em !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 1rem !important;
    }
    
    .mbti-analysis-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .mbti-analysis-card h1 {
        color: white !important;
        font-size: 3em !important;
        margin-bottom: 0.5rem !important;
        font-weight: 900 !important;
        text-align: center;
    }
    
    .mbti-analysis-card h2 {
        color: white !important;
        font-size: 1.8em !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 1rem !important;
    }
    
    .analysis-section {
        background: #1a1a1a !important;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #333333;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.2);
    }
    
    .analysis-section h3 {
        color: #1f77b4 !important;
        font-size: 1.5em !important;
        font-weight: 900 !important;
        margin-bottom: 1rem !important;
    }
    
    .analysis-section p {
        color: #ffffff !important;
        font-size: 1.1em !important;
        line-height: 1.6 !important;
        font-weight: 600 !important;
        margin-bottom: 0.8rem !important;
    }
    
    .selection-info {
        background: #1a1a1a !important;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.2);
    }
    
    .selection-info p {
        color: #ffffff !important;
        font-weight: 900 !important;
        margin: 0 !important;
        font-size: 1.2em !important;
    }
    
    .progress-text {
        font-size: 18px;
        font-weight: 600;
        color: #1f77b4 !important;
        text-align: center;
        margin: 1rem 0;
        background: #1a1a1a !important;
        padding: 1rem;
        border-radius: 10px;
    }
    
    .feature-box {
        text-align: center;
        padding: 1rem;
        background: #1a1a1a !important;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.2);
        margin: 0.5rem;
        border: 2px solid #333333;
    }
    
    .feature-box h4 {
        color: #1f77b4 !important;
        font-weight: 900 !important;
        font-size: 1.2em !important;
    }
    
    .feature-box p {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.1em !important;
    }
    
    .main-title {
        color: #1f77b4 !important;
        font-size: 4em !important;
        margin-bottom: 0.5rem !important;
        font-weight: 900 !important;
        text-align: center;
        text-shadow: 0 0 20px rgba(31, 119, 180, 0.5);
    }
    
    .main-subtitle {
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.5em !important;
        text-align: center;
    }
    
    .stProgress > div > div > div {
        background: #1f77b4 !important;
    }
    
    .stSlider > div > div > div > div {
        background: #1f77b4 !important;
    }
    
    .stSlider > div > div > div {
        background: #333333 !important;
    }
    
    .stDownloadButton > button {
        background: linear-gradient(45deg, #28a745, #20c997) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

def call_mirror_ai(messages: List[Dict]) -> str:
    """Call the AI Mirror with enhanced error handling"""
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "X-Title": "Dual Personality Analysis"
            },
            data=json.dumps({
                "model": "deepseek/deepseek-chat-v3.1:free",
                "messages": messages
            }),
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            raise Exception(f"Mirror connection error: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {e}")
    except Exception as e:
        raise Exception(f"Mirror error: {e}")

def create_dual_chatbot_interface():
    """Create enhanced dual personality mirror chatbot"""
    
    if "chat_history" not in st.session_state:
        # Use MBTI analysis for chatbot persona
        mbti_analysis = st.session_state.get('mbti_analysis', {})
        neural_analysis = st.session_state.get('neural_analysis', {})
        
        if mbti_analysis and mbti_analysis.get('type') in PERSONALITY_TYPES:
            personality_type = mbti_analysis.get('full_type', 'INFP-T')
            type_info = PERSONALITY_TYPES.get(personality_type.split('-')[0], PERSONALITY_TYPES['INFP'])
            
            welcome_msg = f"Hello! I'm your {type_info['name']} inner voice - your {personality_type} personality reflected back to you. I also know you were classified as '{neural_analysis.get('predicted_label', 'Unknown')}' by our neural network analysis. I can share insights combining both perspectives about your personality. What would you like to explore?"
        else:
            welcome_msg = "Hello! I'm your personality mirror, combining insights from both our MBTI and neural network analyses. What would you like to explore about yourself?"
        
        st.session_state.chat_history = [{"role": "assistant", "content": welcome_msg}]
        st.session_state.api_status = "ready"
    
    st.markdown("""
    <div class="chatbot-container">
        <h3 class="chatbot-header">🪞 Your Dual Analysis Personality Mirror</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced mirror status
    if st.session_state.get('api_status', 'ready') == 'ready':
        st.markdown("""
        <div class="mirror-ready">
            🪞 Your dual analysis mirror is ready - reflecting insights from both systems!
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message-user">
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        elif message["role"] == "assistant":
            st.markdown(f"""
            <div class="chat-message-bot">
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        elif message["role"] == "error":
            st.markdown(f"""
            <div class="chat-message-error">
                ⚠️ {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced chat input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask your mirror...", 
            key="chat_input", 
            placeholder="e.g., How do my two analyses compare? What do both systems say about my creativity?",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Ask", key="send_chat", width='stretch')
    
    # Process chat input
    if send_button and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get analyses
        mbti_analysis = st.session_state.get('mbti_analysis', {})
        scenario = st.session_state.get('selected_scenario', SCENARIOS[0])
        
        # Create dual personality prompt
        personality_prompt = create_advanced_personality_prompt(mbti_analysis, scenario)
        
        # Build messages for API
        system_msg = {"role": "system", "content": personality_prompt}
        recent_history = st.session_state.chat_history[-4:]
        api_messages = [system_msg] + [msg for msg in recent_history if msg["role"] in ["user", "assistant"]]
        
        # Call mirror
        with st.spinner("🪞 Your mirror is analyzing both perspectives..."):
            try:
                ai_response = call_mirror_ai(api_messages)
                st.session_state.api_status = "ready"
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                st.session_state.api_status = "error"
                error_msg = f"Mirror is temporarily unavailable. Error: {str(e)}"
                st.session_state.chat_history.append({"role": "error", "content": error_msg})
        
        st.rerun()

def initialize_session_state():
    """Initialize session state variables"""
    if "answers" not in st.session_state:
        st.session_state.answers = {}
    if "assessment_started" not in st.session_state:
        st.session_state.assessment_started = False
    if "results_ready" not in st.session_state:
        st.session_state.results_ready = False
    if "selected_scenario" not in st.session_state:
        st.session_state.selected_scenario = random.choice(SCENARIOS)
    if "adaptive_system" not in st.session_state:
        st.session_state.adaptive_system = DualAnalysisQuestionSystem(st.session_state.selected_scenario)
    if "questions_asked" not in st.session_state:
        st.session_state.questions_asked = []

def create_progress_bar(current: int, total: int) -> None:
    """Create enhanced progress bar that never exceeds 1.0"""
    progress = min(current / total, 1.0) if total > 0 else 0.0
    
    st.markdown(f"""
    <div class="progress-text">
        Question {current + 1} of {total} ({progress:.0%} Complete) - Dual Analysis
    </div>
    """, unsafe_allow_html=True)
    st.progress(progress)

def create_elegant_radar_chart(analysis):
    """Create elegant radar chart for MBTI analysis"""
    
    mbti_scores = analysis['mbti_scores']
    
    categories = ['Extraversion', 'Intuition', 'Feeling', 'Perceiving']
    values = [
        mbti_scores['E/I'],
        10 - mbti_scores['S/N'],
        10 - mbti_scores['T/F'],
        10 - mbti_scores['J/P']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=analysis['type'],
        line=dict(color='#667eea', width=3),
        fillcolor='rgba(102, 126, 234, 0.3)',
        marker=dict(size=8, color='#667eea')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickfont=dict(size=12, color='white'),
                gridcolor='rgba(255, 255, 255, 0.3)',
                linecolor='rgba(255, 255, 255, 0.3)'
            ),
            angularaxis=dict(
                tickfont=dict(size=14, color='white', family='Arial Black'),
                gridcolor='rgba(255, 255, 255, 0.3)',
                linecolor='rgba(255, 255, 255, 0.3)'
            ),
            bgcolor='rgba(26, 26, 26, 0.8)'
        ),
        showlegend=False,
        title=dict(
            text=f"MBTI Analysis: {analysis['type']}",
            font=dict(size=20, color='white', family='Arial Black'),
            x=0.5,
            y=0.95
        ),
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='white', size=12),
        width=600,
        height=500,
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig

def create_neural_probability_chart(analysis):
    """Create probability distribution chart for neural network analysis"""
    
    labels = analysis['labels']
    probabilities = analysis['probabilities']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=probabilities * 100,
            marker_color=['#e74c3c' if label == analysis['predicted_label'] else '#c0392b' for label in labels],
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Neural Network Probability Distribution",
        title_font_color="white",
        xaxis_title="Personality Types",
        yaxis_title="Confidence (%)",
        xaxis_title_font_color="white",
        yaxis_title_font_color="white",
        xaxis_tickfont_color="white",
        yaxis_tickfont_color="white",
        yaxis=dict(range=[0, 100]),
        height=500,
        template="plotly_dark",
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a'
    )
    
    return fig

def create_dual_results_visualization(neural_analysis, mbti_analysis, scenario):
    """Create comprehensive dual results visualization"""
    
    st.markdown("""
    <h1 style="color: white; text-align: center; font-size: 2.5em; margin: 2rem 0;">
        🔬 Dual Personality Analysis Results
    </h1>
    """, unsafe_allow_html=True)
    
    # Dual analysis cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="neural-analysis-card">
            <h1>🧠 Neural Network</h1>
            <h2>{neural_analysis['predicted_label']}</h2>
            <p>Confidence: {neural_analysis['confidence']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Neural network probability chart
        fig_neural = create_neural_probability_chart(neural_analysis)
        st.plotly_chart(fig_neural, width='stretch')
    
    with col2:
        type_info = PERSONALITY_TYPES.get(mbti_analysis['type'], PERSONALITY_TYPES['INFP'])
        
        st.markdown(f"""
        <div class="mbti-analysis-card">
            <h1>🎯 MBTI System</h1>
            <h2>{mbti_analysis['full_type']} - {type_info['name']}</h2>
            <p>{type_info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # MBTI radar chart
        fig_mbti = create_elegant_radar_chart(mbti_analysis)
        st.plotly_chart(fig_mbti, width='stretch')
    
    # Dual chatbot interface
    create_dual_chatbot_interface()
    
    # Comparative analysis
    st.markdown("""
    <h2 style="color: white; font-size: 2em; margin: 2rem 0 1rem 0;">
        📊 Comparative Analysis
    </h2>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="analysis-section">
            <h3>🧠 Neural Network Insights</h3>
            <p><strong>Primary Classification:</strong> {neural_analysis['predicted_label']}</p>
            <p><strong>Confidence Level:</strong> {neural_analysis['confidence']:.1%}</p>
            <p><strong>Analysis Method:</strong> Deep learning on 29 behavioral features</p>
            <p><strong>Top Alternatives:</strong> {', '.join([f"{label} ({prob:.1%})" for label, prob in zip(neural_analysis['labels'], neural_analysis['probabilities']) if prob > 0.1 and label != neural_analysis['predicted_label']][:2])}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="analysis-section">
            <h3>🎯 MBTI System Insights</h3>
            <p><strong>Primary Type:</strong> {mbti_analysis['full_type']} - {type_info['name']}</p>
            <p><strong>Core Strengths:</strong> {type_info['strengths'][:100]}...</p>
            <p><strong>Growth Areas:</strong> {type_info['weaknesses'][:100]}...</p>
            <p><strong>Career Matches:</strong> {type_info['careers'][:100]}...</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature analysis comparison
    st.markdown("""
    <h2 style="color: white; font-size: 2em; margin: 2rem 0 1rem 0;">
        🔍 Key Insights Comparison
    </h2>
    """, unsafe_allow_html=True)
    
    comparison_df = pd.DataFrame({
        'Analysis System': ['Neural Network', 'MBTI System'],
        'Primary Result': [neural_analysis['predicted_label'], f"{mbti_analysis['full_type']} - {type_info['name']}"],
        'Confidence': [f"{neural_analysis['confidence']:.1%}", f"{mbti_analysis['confidence']:.1%}"],
        'Method': ['Deep learning on 29 features', '4-dimension MBTI scoring'],
        'Focus': ['Learned patterns from data', 'Psychological theory-based']
    })
    
    st.dataframe(comparison_df, width='stretch')

def main():
    """Enhanced main application with dual analysis system"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="main-title">🔬 Dual Personality Analysis System</h1>
        <h3 class="main-subtitle">Neural Network + MBTI Combined Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.assessment_started:
        # Display selected scenario
        scenario = st.session_state.selected_scenario
        
        st.markdown(f"""
        <div class="scenario-box">
            <h2>🌟 {scenario['name']}</h2>
            <p>{scenario['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Instructions
        st.markdown("""
        **Get insights from TWO advanced systems: Your responses will be analyzed by both our custom neural network (trained on behavioral patterns) AND our MBTI system (based on psychological theory). You'll receive dual perspectives on your personality!**
        """)
        
        # Enhanced features
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="feature-box">
                <h4>🧠 Neural Network</h4>
                <p>Deep learning analysis</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="feature-box">
                <h4>🎯 MBTI System</h4>
                <p>16 personality types</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="feature-box">
                <h4>🪞 Dual Mirror</h4>
                <p>Combined insights AI</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🚀 Begin Dual Analysis", key="start_btn"):
            st.session_state.assessment_started = True
            st.rerun()
    
    elif not st.session_state.results_ready:
        # Dual analysis question system
        adaptive_system = st.session_state.adaptive_system
        
        # Get next question
        next_question = adaptive_system.get_next_question(st.session_state.questions_asked, st.session_state.answers)
        
        # Check if analysis should complete
        if next_question is None or not adaptive_system.should_continue_asking(st.session_state.questions_asked):
            # Run both analyses
            with st.spinner("🔬 Running dual personality analysis - Neural Network + MBTI..."):
                # Load models
                model, scaler, label_encoder, device = load_and_train_dual_models()
                
                # Map answers to neural network features
                feature_vector = map_answers_to_neural_features(st.session_state.answers)
                
                # Run neural network analysis
                neural_analysis = analyze_neural_network(feature_vector, model, scaler, label_encoder, device)
                
                # Run MBTI analysis
                mbti_analysis = analyze_mbti_system(st.session_state.answers)
                
                # Store both analyses
                st.session_state.neural_analysis = neural_analysis
                st.session_state.mbti_analysis = mbti_analysis
                st.session_state.results_ready = True
                st.rerun()
        else:
            # Show progress
            target_questions = adaptive_system.target_question_count
            current_questions = len(st.session_state.questions_asked)
            
            create_progress_bar(current_questions, target_questions)
            
            # Display question
            st.markdown(f"""
            <div class="question-box">
                <h3>Question {current_questions + 1}</h3>
                <p>{next_question['text']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            current_answer = st.session_state.answers.get(next_question['id'], 5)
            
            answer = st.slider(
                "Your Response",
                min_value=1,
                max_value=10,
                value=current_answer,
                key=f"dual_slider_{next_question['id']}_{current_questions}",
                help="1 = Strongly Disagree | 5 = Neutral | 10 = Strongly Agree"
            )
            
            st.session_state.answers[next_question['id']] = answer
            
            labels = ["Strongly Disagree", "Disagree", "Somewhat Disagree", "Slightly Disagree", 
                     "Neutral", "Slightly Agree", "Somewhat Agree", "Agree", "Strongly Agree", "Completely Agree"]
            
            st.markdown(f"""
            <div class="selection-info">
                <p><strong>Your response:</strong> {answer}/10 - {labels[answer-1]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Navigation
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if len(st.session_state.questions_asked) > 0:
                    if st.button("⬅️ Previous", key="prev_btn"):
                        if st.session_state.questions_asked:
                            last_question = st.session_state.questions_asked.pop()
                        st.rerun()
            
            with col3:
                if st.button("Next ➡️", key="next_btn"):
                    if next_question['id'] not in st.session_state.questions_asked:
                        st.session_state.questions_asked.append(next_question['id'])
                    st.rerun()
    
    else:
        # Display dual results
        neural_analysis = st.session_state.neural_analysis
        mbti_analysis = st.session_state.mbti_analysis
        scenario = st.session_state.selected_scenario
        
        create_dual_results_visualization(neural_analysis, mbti_analysis, scenario)
        
        # Download comprehensive results
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retake Assessment", key="retake_btn"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col2:
            type_info = PERSONALITY_TYPES.get(mbti_analysis['type'], PERSONALITY_TYPES['INFP'])
            
            results_text = f"""Dual Personality Analysis Results
========================================

NEURAL NETWORK ANALYSIS:
Primary Classification: {neural_analysis['predicted_label']}
Confidence: {neural_analysis['confidence']:.1%}
Method: Deep learning on 29 behavioral features
Alternative Classifications: {', '.join([f"{label} ({prob:.1%})" for label, prob in zip(neural_analysis['labels'], neural_analysis['probabilities']) if prob > 0.05 and label != neural_analysis['predicted_label']])}

MBTI SYSTEM ANALYSIS:
Primary Type: {mbti_analysis['full_type']} - {type_info['name']}
Description: {type_info['description']}
Core Strengths: {type_info['strengths']}
Growth Areas: {type_info['weaknesses']}
Career Matches: {type_info['careers']}
Confidence: {mbti_analysis['confidence']:.1%}

MBTI DIMENSION SCORES:
Extraversion/Introversion: {mbti_analysis['mbti_scores']['E/I']:.1f}/10
Sensing/Intuition: {mbti_analysis['mbti_scores']['S/N']:.1f}/10  
Thinking/Feeling: {mbti_analysis['mbti_scores']['T/F']:.1f}/10
Judging/Perceiving: {mbti_analysis['mbti_scores']['J/P']:.1f}/10

SCENARIO: {scenario['name']}
QUESTIONS ANALYZED: {len(st.session_state.questions_asked)} adaptive questions

This dual analysis provides both data-driven insights (neural network) and theory-based understanding (MBTI) of your personality patterns.
            """
            st.download_button(
                label="📄 Download Dual Analysis",
                data=results_text,
                file_name=f"dual_personality_analysis_{neural_analysis['predicted_label']}_{mbti_analysis['full_type']}.txt",
                mime="text/plain"
            )

footer = """
<style>
.custom-footer {
  position: fixed;
  left: 0; bottom: 0; width: 100%;
  padding: 8px 12px;
  text-align: center;
  font-weight: 800; font-size: 12px;
  z-index: 9999;
  background: linear-gradient(90deg,#1a1a1a, #2a2a2a);
  color: #17becf;
  border-top: 2px solid #1f77b4;
  box-shadow: 0 -2px 10px rgba(23,190,207,0.15);
}
</style>
<div class="custom-footer">Made by Aariv</div>
"""
st.markdown(footer, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
