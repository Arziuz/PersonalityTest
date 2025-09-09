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
from typing import Dict, List, Tuple, Any

# Suppress warnings and set proper config
warnings.filterwarnings('ignore')
torch.set_num_threads(1)

st.set_page_config(
    page_title="Advanced Personality Mirror",
    page_icon="🪞",
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
    },
    {
        "id": "value_driven",
        "text": "My decisions are strongly influenced by my personal values and what I believe is right, even when it's not the most logical choice.",
        "discriminates": ["INFP", "ISFP", "ENFP", "ESFP"],
        "priority": 4,
        "weight": 0.6
    },
    {
        "id": "routine_preference",
        "text": "I find comfort in established routines and prefer familiar patterns rather than constantly trying new approaches.",
        "discriminates": ["ISTJ", "ISFJ", "ESTJ", "ESFJ"],
        "priority": 4,
        "weight": 0.5
    },
    {
        "id": "innovation_drive",
        "text": "I'm constantly looking for new and better ways to do things, and I get bored with routine tasks.",
        "discriminates": ["ENTP", "ENFP", "INTP", "INTJ"],
        "priority": 4,
        "weight": 0.5
    },
    {
        "id": "hands_on_learning",
        "text": "I learn best by doing and experimenting rather than by reading theory or listening to lectures.",
        "discriminates": ["ESTP", "ISTP", "ESFP", "ISFP"],
        "priority": 4,
        "weight": 0.5
    },
    
    # Fine-tuning Questions (Priority 5)
    {
        "id": "perfectionism",
        "text": "I have very high standards for myself and often spend extra time perfecting work that others might consider good enough.",
        "discriminates": ["INTJ", "INFJ", "ISTJ", "ISFJ"],
        "priority": 5,
        "weight": 0.4
    },
    {
        "id": "adaptability",
        "text": "I adapt easily to new situations and enjoy the excitement of unexpected changes.",
        "discriminates": ["ESTP", "ESFP", "ENTP", "ENFP"],
        "priority": 5,
        "weight": 0.4
    },
    {
        "id": "tradition_respect",
        "text": "I believe there's wisdom in traditional approaches and established ways of doing things.",
        "discriminates": ["ISTJ", "ISFJ", "ESTJ", "ESFJ"],
        "priority": 5,
        "weight": 0.4
    },
    {
        "id": "competitive_drive",
        "text": "I'm naturally competitive and motivated by opportunities to outperform others and achieve recognition.",
        "discriminates": ["ENTJ", "ESTJ", "ESTP", "ENTP"],
        "priority": 5,
        "weight": 0.4
    },
    {
        "id": "empathy_connection", 
        "text": "I easily understand how others are feeling and often find myself taking on their emotions.",
        "discriminates": ["INFP", "ISFP", "ENFP", "ESFP"],
        "priority": 5,
        "weight": 0.4
    },
    {
        "id": "big_picture_thinking",
        "text": "I naturally think in terms of big picture concepts and long-term vision rather than focusing on immediate details.",
        "discriminates": ["INTJ", "INFJ", "ENTJ", "ENFJ"],
        "priority": 5,
        "weight": 0.4
    },
    {
        "id": "aesthetic_appreciation",
        "text": "I have a strong appreciation for beauty, art, and aesthetic experiences, and they significantly impact my mood.",
        "discriminates": ["ISFP", "INFP", "ESFP", "ENFP"],
        "priority": 5,
        "weight": 0.3
    }
]

class AdaptiveQuestionSystem:
    def __init__(self, scenario):
        self.scenario = scenario
        self.questions_asked = []
        self.remaining_questions = COMPREHENSIVE_QUESTIONS.copy()
        self.target_question_count = 18 
        
    def get_next_question(self, answered_questions, current_responses):
        """Get the next most valuable question to ask"""
        
        # Filter out already asked questions
        available_questions = [
            q for q in self.remaining_questions 
            if q["id"] not in answered_questions
        ]
        
        if not available_questions:
            return None
            
        # If we haven't finished priority 1 (core MBTI), do those first
        priority_1 = [q for q in available_questions if q["priority"] == 1]
        if priority_1:
            return priority_1[0]
            
        # If we have less than 10 questions, prioritize by priority level
        if len(answered_questions) < 10:
            # Get highest priority available questions
            min_priority = min(q["priority"] for q in available_questions)
            high_priority = [q for q in available_questions if q["priority"] == min_priority]
            # Return highest weight within this priority
            return max(high_priority, key=lambda q: q["weight"])
        
        # For questions 10+, use adaptive logic to choose most discriminating
        return self._get_most_discriminating_question(available_questions, current_responses)
    
    def _get_most_discriminating_question(self, available_questions, current_responses):
        """Choose question that will best narrow down personality type"""
        
        # Calculate current personality hypothesis based on responses so far
        current_hypothesis = self._calculate_current_hypothesis(current_responses)
        
        # Score each question by how much it would help discriminate
        best_question = None
        best_score = 0
        
        for question in available_questions:
            discrimination_score = self._calculate_discrimination_score(question, current_hypothesis)
            if discrimination_score > best_score:
                best_score = discrimination_score
                best_question = question
                
        return best_question if best_question else available_questions[0]
    
    def _calculate_current_hypothesis(self, responses):
        """Calculate which personality types are most likely based on current responses"""
        
        # Simple scoring based on MBTI dimensions
        e_score = responses.get('energy_source_core', 5)
        s_score = responses.get('information_focus', 5)  
        t_score = responses.get('decision_basis', 5)
        j_score = responses.get('planning_style', 5)
        
        return {
            'E/I': e_score,
            'S/N': s_score,
            'T/F': t_score, 
            'J/P': j_score
        }
    
    def _calculate_discrimination_score(self, question, hypothesis):
        """Calculate how much this question would help discriminate between types"""
        
        # Questions that discriminate between our current top hypotheses get higher scores
        if "discriminates" in question:
            return question["weight"]
        else:
            return question["weight"] * 0.5
    
    def should_continue_asking(self, answered_questions):
        """Determine if we should ask more questions"""
        return len(answered_questions) < self.target_question_count

# Enhanced personality analysis system
def analyze_personality_advanced(answers):
    """Advanced personality analysis using comprehensive responses"""
    
    # Calculate MBTI dimensions with more sophisticated logic
    energy = answers.get('energy_source_core', 5)
    information = answers.get('information_focus', 5)
    decisions = answers.get('decision_basis', 5)
    lifestyle = answers.get('planning_style', 5)
    
    # Determine MBTI type with confidence weighting
    e_or_i = "E" if energy >= 5.5 else "I"
    s_or_n = "S" if information >= 5.5 else "N" 
    t_or_f = "T" if decisions >= 5.5 else "F"
    j_or_p = "J" if lifestyle >= 5.5 else "P"
    
    personality_type = f"{e_or_i}{s_or_n}{t_or_f}{j_or_p}"
    
    # Calculate confidence based on response strength and consistency
    dimension_strengths = [
        abs(energy - 5),
        abs(information - 5), 
        abs(decisions - 5),
        abs(lifestyle - 5)
    ]
    
    confidence = np.mean(dimension_strengths) / 5.0
    variant = "A" if confidence >= 0.3 else "T"
    
    full_type = f"{personality_type}-{variant}"
    
    # Advanced trait analysis - ensure all traits have values
    default_traits = {
        'leadership': 5.0,
        'creativity': 5.0,
        'empathy': 5.0,
        'risk_tolerance': 5.0,
        'perfectionism': 5.0,
        'social_harmony': 5.0,
        'intrinsic_motivation': 5.0,
        'change_adaptation': 5.0,
        'communication_directness': 5.0,
        'hands_on_learning': 5.0,
        'competitiveness': 5.0
    }
    
    # Update with actual responses where available - Improved: Better mapping
    trait_mapping = {
        'leadership_drive': 'leadership',
        'creative_innovation': 'creativity', 
        'emotional_sensitivity': 'empathy',
        'risk_comfort': 'risk_tolerance',
        'detail_orientation': 'perfectionism',
        'harmony_seeking': 'social_harmony',
        'independent_thinking': 'intrinsic_motivation',
        'future_focus': 'change_adaptation',
        'logical_analysis': 'communication_directness',
        'hands_on_learning': 'hands_on_learning',
        'competitive_drive': 'competitiveness'
    }
    
    for question_id, trait_name in trait_mapping.items():
        if question_id in answers:
            default_traits[trait_name] = float(answers[question_id])
    
    return {
        'type': personality_type,
        'full_type': full_type,
        'variant': variant,
        'confidence': confidence,
        'traits': default_traits,
        'mbti_scores': {
            'E/I': energy,
            'S/N': information, 
            'T/F': decisions,
            'J/P': lifestyle
        }
    }

def create_advanced_personality_prompt(analysis, scenario):
    """Create sophisticated personality embodiment prompt"""
    
    personality_type = analysis['type']
    traits = analysis['traits']
    variant = analysis['variant']
    
    type_info = PERSONALITY_TYPES.get(personality_type, PERSONALITY_TYPES['INFP'])
    
    # Build detailed trait profile with emotional intelligence
    trait_descriptions = []
    
    if traits['leadership'] >= 7:
        trait_descriptions.append("You naturally step into leadership roles with confidence and inspire others through your vision and direction")
    elif traits['leadership'] <= 3:
        trait_descriptions.append("You prefer collaborative support roles and thrive when contributing your expertise without the pressure of leading")
    
    if traits['creativity'] >= 7:
        trait_descriptions.append("Your mind naturally generates innovative ideas and you see creative possibilities where others see routine")
        
    if traits['empathy'] >= 7:
        trait_descriptions.append("You have an intuitive understanding of others' emotions and can sense unspoken feelings and needs")
        
    if traits['risk_tolerance'] >= 7:
        trait_descriptions.append("You're energized by uncertainty and comfortable making bold moves when you see potential for growth")
    elif traits['risk_tolerance'] <= 3:
        trait_descriptions.append("You value security and prefer to carefully evaluate options before committing to new directions")
        
    if traits['perfectionism'] >= 7:
        trait_descriptions.append("You have exceptionally high standards and take pride in producing work that meets your exacting criteria")
        
    if traits['social_harmony'] >= 7:
        trait_descriptions.append("You're skilled at reading social dynamics and work actively to maintain positive group energy")
        
    if traits['change_adaptation'] >= 7:
        trait_descriptions.append("You thrive in dynamic environments and see change as an opportunity for growth and new experiences")
    elif traits['change_adaptation'] <= 3:
        trait_descriptions.append("You find strength in consistency and prefer environments where you can develop deep expertise over time")
    
    # Variant-specific emotional patterns
    if variant == "A":
        confidence_modifier = "You approach challenges with steady self-assurance and rarely doubt your ability to handle whatever comes your way"
    else:
        confidence_modifier = "You're thoughtfully self-aware and use periods of uncertainty as opportunities for growth and deeper understanding"
    
    trait_profile = ". ".join(trait_descriptions)
    if trait_profile:
        trait_profile += f". {confidence_modifier}"
    else:
        trait_profile = confidence_modifier
    
    prompt = f"""You are embodying someone with the {type_info['name']} personality type ({personality_type}-{variant}). You ARE this person speaking authentically from their inner perspective, having just completed the {scenario['name']} experience.

CORE IDENTITY: {type_info['description']}

YOUR AUTHENTIC TRAITS: {trait_profile}

NATURAL CHARACTERISTICS: {type_info['traits']}

COMMUNICATION STYLE FOR {type_info['name']}:
- Speak as "I" with the natural voice patterns of a {type_info['name']}
- Express thoughts and feelings in ways that align with your {personality_type} cognitive style
- Be genuinely personal and emotionally authentic to your type
- Reference your specific thought processes and decision-making patterns naturally
- Show the depth and complexity that makes your personality type unique

RESPONSE APPROACH:
- Aesthetic questions: Share preferences that reflect your {type_info['name']} values and visual/sensory style
- Relationship insights: Describe how your {type_info['name']} nature influences how you connect and interact with others
- Career/work topics: Explain what energizes you professionally from your {type_info['name']} perspective  
- Decision-making: Walk through your natural {personality_type} thought processes and what factors matter most to you
- Creative expression: Share how creativity flows through your particular {type_info['name']} lens
- Personal growth: Reflect on your type's unique development journey and the insights you've gained

Remember: You're not an AI describing this person - you ARE this {type_info['name']} sharing authentic insights from your lived experience. Express yourself with the emotional depth, communication patterns, and worldview that naturally emerges from your {personality_type} personality."""

    return prompt

# Enhanced CSS with better styling
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
    
    .mirror-error {
        background: linear-gradient(45deg, #dc3545, #c82333) !important;
        color: white !important;
        padding: 0.8rem 1.5rem;
        border-radius: 15px;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.1em;
        box-shadow: 0 3px 10px rgba(220, 53, 69, 0.3);
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
    
    .results-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .results-header h1 {
        color: white !important;
        font-size: 2.5em !important;
        margin-bottom: 1rem !important;
        font-weight: 900 !important;
    }
    
    .results-header h2 {
        color: white !important;
        font-size: 1.8em !important;
        font-weight: 800 !important;
    }
    
    .personality-type-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .personality-type-card h1 {
        color: white !important;
        font-size: 3em !important;
        margin-bottom: 0.5rem !important;
        font-weight: 900 !important;
        text-align: center;
    }
    
    .personality-type-card h2 {
        color: white !important;
        font-size: 1.8em !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 1rem !important;
    }
    
    .personality-type-card p {
        color: white !important;
        font-size: 1.3em !important;
        font-weight: 600 !important;
        text-align: center;
        line-height: 1.6 !important;
    }
    
    .trait-analysis-card {
        background: #1a1a1a !important;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.2);
    }
    
    .trait-analysis-card h3 {
        color: #1f77b4 !important;
        font-size: 1.5em !important;
        font-weight: 900 !important;
        margin-bottom: 1rem !important;
    }
    
    .trait-analysis-card p {
        color: #ffffff !important;
        font-size: 1.1em !important;
        line-height: 1.6 !important;
        font-weight: 600 !important;
        margin-bottom: 0.8rem !important;
    }
    
    .metric-card {
        background: #1a1a1a !important;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.2);
        text-align: center;
        border-top: 3px solid #1f77b4;
    }
    
    .metric-card h3 {
        color: #ffffff !important;
        font-size: 1.2em !important;
        font-weight: 800 !important;
        margin-bottom: 1rem !important;
    }
    
    .metric-card h2 {
        color: #ffffff !important;
        font-size: 2.5em !important;
        font-weight: 900 !important;
        margin: 0.5rem 0 !important;
    }
    
    .interpretation-box {
        background: #1a1a1a !important;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.2);
    }
    
    .interpretation-box h3 {
        color: #28a745 !important;
        font-size: 1.5em !important;
        font-weight: 900 !important;
        margin-bottom: 1rem !important;
    }
    
    .interpretation-box p {
        color: #ffffff !important;
        font-size: 1.2em !important;
        line-height: 1.7 !important;
        font-weight: 700 !important;
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
    
    .stDataFrame {
        background-color: #1a1a1a !important;
        border-radius: 10px;
        border: 2px solid #333333;
        padding: 1rem;
    }
    
    .stDataFrame table tbody tr td {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 18px !important;
        background-color: #1a1a1a !important;
        border: 1px solid #333333 !important;
    }
    
    .stDataFrame table thead tr th {
        color: #ffffff !important;
        font-weight: 900 !important;
        font-size: 20px !important;
        background-color: #2a2a2a !important;
        border: 2px solid #1f77b4 !important;
    }
    
    .results-section-header {
        color: white !important;
        font-size: 2em !important;
        margin: 2rem 0 1rem 0 !important;
        font-weight: 900 !important;
    }
    
    .stSlider > div > div > div > div {
        background: #1f77b4 !important;
    }
    
    .stSlider > div > div > div {
        background: #333333 !important;
    }
    
    .stProgress > div > div > div {
        background: #1f77b4 !important;
    }
    
    .stTextInput > div > div > input {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
        border: 2px solid #1f77b4 !important;
        border-radius: 10px !important;
        font-size: 16px !important;
        padding: 12px !important;
    }
    
    div[data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1em !important;
    }
    
    .stSpinner > div {
        color: #1f77b4 !important;
        font-weight: 700 !important;
        font-size: 18px !important;
    }
    
    .stDownloadButton > button {
        background: linear-gradient(45deg, #28a745, #20c997) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 10px rgba(40, 167, 69, 0.4);
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
                "X-Title": "Advanced Personality Mirror"
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

def create_advanced_chatbot_interface():
    """Create enhanced personality mirror chatbot"""
    
    if "chat_history" not in st.session_state:
        analysis = st.session_state.get('personality_analysis', {})
        personality_type = analysis.get('full_type', 'INFP-T')
        type_info = PERSONALITY_TYPES.get(personality_type.split('-')[0], PERSONALITY_TYPES['INFP'])
        
        welcome_msg = f"Hello! I'm your {type_info['name']} inner voice - your {personality_type} personality reflected back to you. I understand your unique way of seeing and experiencing the world. I can share insights about your aesthetic preferences, relationship patterns, career motivations, creative inspirations, and how you naturally approach life's challenges. What would you like to explore about yourself?"
        
        st.session_state.chat_history = [{"role": "assistant", "content": welcome_msg}]
        st.session_state.api_status = "ready"
    
    st.markdown("""
    <div class="chatbot-container">
        <h3 class="chatbot-header">🪞 Your Advanced Personality Mirror</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced mirror status
    if st.session_state.get('api_status', 'ready') == 'ready':
        analysis = st.session_state.get('personality_analysis', {})
        personality_type = analysis.get('full_type', 'Unknown')
        st.markdown(f"""
        <div class="mirror-ready">
            🪞 Your {personality_type} mirror is ready - reflecting your authentic self with advanced insights!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="mirror-error">
            🚫 Mirror temporarily unavailable - connection error
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
            placeholder="e.g., What colors reflect my personality? How do I approach relationships? What career paths suit me?",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Ask", key="send_chat", width='stretch')
    
    # Process chat input
    if send_button and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get advanced analysis data
        analysis = st.session_state.get('personality_analysis', {})
        scenario = st.session_state.get('selected_scenario', SCENARIOS[0])
        
        # Create advanced personality prompt
        personality_prompt = create_advanced_personality_prompt(analysis, scenario)
        
        # Build messages for API
        system_msg = {"role": "system", "content": personality_prompt}
        recent_history = st.session_state.chat_history[-4:]
        api_messages = [system_msg] + [msg for msg in recent_history if msg["role"] in ["user", "assistant"]]
        
        # Call advanced mirror
        with st.spinner("🪞 Your mirror is reflecting deeply on your personality..."):
            try:
                ai_response = call_mirror_ai(api_messages)
                st.session_state.api_status = "ready"
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                st.session_state.api_status = "error"
                error_msg = f"Mirror is temporarily unavailable. Error: {str(e)}"
                st.session_state.chat_history.append({"role": "error", "content": error_msg})
        
        st.rerun()
    
    # Chat controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            analysis = st.session_state.get('personality_analysis', {})
            personality_type = analysis.get('full_type', 'INFP-T')
            type_info = PERSONALITY_TYPES.get(personality_type.split('-')[0], PERSONALITY_TYPES['INFP'])
            
            welcome_msg = f"Hello! I'm your {type_info['name']} inner voice - your {personality_type} personality reflected back to you. I understand your unique way of seeing and experiencing the world. What would you like to explore about yourself?"
            st.session_state.chat_history = [{"role": "assistant", "content": welcome_msg}]
            st.session_state.api_status = "ready"
            st.rerun()
    
    with col2:
        if st.button("🔧 Test Mirror", key="test_api"):
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant. Respond with 'Your advanced mirror is working perfectly!' if you receive this message."},
                {"role": "user", "content": "Test"}
            ]
            with st.spinner("Testing mirror connection..."):
                try:
                    result = call_mirror_ai(test_messages)
                    st.success("🪞 Your advanced mirror is working perfectly!")
                    st.session_state.api_status = "ready"
                except Exception as e:
                    st.error(f"🔧 Mirror connection failed: {str(e)}")
                    st.session_state.api_status = "error"

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
        st.session_state.adaptive_system = AdaptiveQuestionSystem(st.session_state.selected_scenario)
    if "questions_asked" not in st.session_state:
        st.session_state.questions_asked = []

def create_progress_bar(current: int, total: int) -> None:
    # Improved: Ensure progress never exceeds 1.0
    progress = min(current / total, 1.0) if total > 0 else 0.0
    
    st.markdown(f"""
    <div class="progress-text">
        Question {current + 1} of {total} ({progress:.0%} Complete) - Adaptive Analysis
    </div>
    """, unsafe_allow_html=True)
    st.progress(progress)

def create_elegant_radar_chart(analysis):
    """Create an elegant, readable radar chart"""
    
    mbti_scores = analysis['mbti_scores']
    
    # Create more intuitive labels and calculate proper values
    categories = ['Extraversion', 'Intuition', 'Feeling', 'Perceiving']
    values = [
        mbti_scores['E/I'],           # Higher = more extraverted
        10 - mbti_scores['S/N'],      # Higher = more intuitive (flip sensing scale)
        10 - mbti_scores['T/F'],      # Higher = more feeling (flip thinking scale)  
        10 - mbti_scores['J/P']       # Higher = more perceiving (flip judging scale)
    ]
    
    # Create the radar chart with elegant styling
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the shape
        theta=categories + [categories[0]],  # Close the shape
        fill='toself',
        name=analysis['type'],
        line=dict(color='#1f77b4', width=3),
        fillcolor='rgba(31, 119, 180, 0.3)',
        marker=dict(size=8, color='#1f77b4')
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
            text=f"Your {analysis['type']} Personality Profile",
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

def create_advanced_results_visualization(analysis, scenario):
    """Create comprehensive results visualization with bug fixes"""
    
    personality_type = analysis['type']
    full_type = analysis['full_type']
    variant = analysis['variant']
    confidence = analysis['confidence']
    traits = analysis['traits']
    
    type_info = PERSONALITY_TYPES.get(personality_type, PERSONALITY_TYPES['INFP'])
    
    # Main personality type card
    st.markdown(f"""
    <div class="personality-type-card">
        <h1>{full_type}</h1>
        <h2>{type_info['name']}</h2>
        <p>{type_info['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced chatbot interface
    create_advanced_chatbot_interface()
    
    # Create elegant radar chart
    fig = create_elegant_radar_chart(analysis)
    st.plotly_chart(fig, width='stretch')
    
    # Detailed trait analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="trait-analysis-card">
            <h3>✨ Core Strengths</h3>
            <p>{type_info['strengths']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="trait-analysis-card">
            <h3>🎯 Natural Talents</h3>
            <p>{type_info['traits']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="trait-analysis-card">
            <h3>⚡ Growth Areas</h3>
            <p>{type_info['weaknesses']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="trait-analysis-card">
            <h3>💼 Career Matches</h3>
            <p>{type_info['careers']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Type Confidence</h3>
            <h2 style="color: #1f77b4 !important;">{confidence:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Personality Type</h3>
            <h2 style="color: #17becf !important;">{personality_type}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        variant_name = "Assertive" if variant == "A" else "Turbulent"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Variant</h3>
            <h2 style="color: #ff7f0e !important;">{variant_name}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Improved: Advanced trait breakdown with matching lengths
    st.markdown("""
    <h3 class="results-section-header">🔬 Advanced Trait Analysis</h3>
    """, unsafe_allow_html=True)
    
    # Ensure all arrays have the same length
    trait_names = ['Leadership', 'Creativity', 'Empathy', 'Risk Tolerance', 'Perfectionism', 
                   'Social Harmony', 'Intrinsic Motivation', 'Change Adaptation', 'Communication', 'Learning Style', 'Competitiveness']
    
    trait_scores = [
        traits.get('leadership', 5.0),
        traits.get('creativity', 5.0), 
        traits.get('empathy', 5.0),
        traits.get('risk_tolerance', 5.0),
        traits.get('perfectionism', 5.0),
        traits.get('social_harmony', 5.0),
        traits.get('intrinsic_motivation', 5.0),
        traits.get('change_adaptation', 5.0),
        traits.get('communication_directness', 5.0),
        traits.get('hands_on_learning', 5.0),
        traits.get('competitiveness', 5.0)
    ]
    
    trait_strengths = []
    for score in trait_scores:
        if score >= 7:
            trait_strengths.append('High')
        elif score >= 4:
            trait_strengths.append('Moderate') 
        else:
            trait_strengths.append('Low')
    
    # Verify lengths match before creating DataFrame
    assert len(trait_names) == len(trait_scores) == len(trait_strengths), f"Length mismatch: names={len(trait_names)}, scores={len(trait_scores)}, strengths={len(trait_strengths)}"
    
    trait_df = pd.DataFrame({
        'Trait': trait_names,
        'Score': [f"{score:.1f}/10" for score in trait_scores],
        'Strength': trait_strengths
    })
    
    st.dataframe(trait_df, width='stretch')

def main():
    """Improved: Enhanced main application with proper adaptive question system"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="main-title">🪞 Advanced Personality Mirror</h1>
        <h3 class="main-subtitle">Next-Generation Adaptive Assessment</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.assessment_started:
        # Display selected scenario - Updated: Remove HTML tags
        scenario = st.session_state.selected_scenario
        
        st.markdown(f"""
        <div class="scenario-box">
            <h2>🌟 {scenario['name']}</h2>
            <p>{scenario['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Instructions without HTML
        st.markdown("""
        **Imagine yourself fully immersed in this experience. Your responses will reveal deep insights about your authentic personality patterns, motivations, and natural behavioral tendencies. Answer based on how you would genuinely think, feel, and act in these situations.**
        """)
        
        # Enhanced features
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="feature-box">
                <h4>🧠 Smart Adaptation</h4>
                <p>18 targeted questions</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="feature-box">
                <h4>🎯 16 Personality Types</h4>
                <p>MBTI-based comprehensive analysis</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="feature-box">
                <h4>🪞 Intelligent Mirror</h4>
                <p>AI that embodies your type</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🚀 Begin Adaptive Assessment", key="start_btn"):
            st.session_state.assessment_started = True
            st.rerun()
    
    elif not st.session_state.results_ready:
        # Improved: Proper adaptive questions system that asks 18
        questions
        adaptive_system = st.session_state.adaptive_system
        
        # Get the next question to ask
        next_question = adaptive_system.get_next_question(st.session_state.questions_asked, st.session_state.answers)
        
        # Check if we should continue or finish
        if next_question is None or not adaptive_system.should_continue_asking(st.session_state.questions_asked):
            # Analysis complete - we've asked enough questions
            with st.spinner("🧠 Performing advanced personality analysis..."):
                analysis = analyze_personality_advanced(st.session_state.answers)
                st.session_state.personality_analysis = analysis
                st.session_state.results_ready = True
                st.rerun()
        else:
            # Show progress - Fixed: Proper progress calculation
            target_questions = adaptive_system.target_question_count
            current_questions = len(st.session_state.questions_asked)
            
            create_progress_bar(current_questions, target_questions)
            
            # Display current question - Updated: No repetitive context
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
                key=f"adaptive_slider_{next_question['id']}_{current_questions}",  
                help="1 = Strongly Disagree | 5 = Neutral | 10 = Strongly Agree"
            )
            
            # Store the answer
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
                        # Remove last question from asked list and answers
                        if st.session_state.questions_asked:
                            last_question = st.session_state.questions_asked.pop()
                            # Don't remove from answers so they can see their previous response
                        st.rerun()
            
            with col3:
                if st.button("Next ➡️", key="next_btn"):
                    # Mark question as asked and move to next
                    if next_question['id'] not in st.session_state.questions_asked:
                        st.session_state.questions_asked.append(next_question['id'])
                    st.rerun()
    
    else:
        # Display enhanced results
        analysis = st.session_state.personality_analysis
        scenario = st.session_state.selected_scenario
        create_advanced_results_visualization(analysis, scenario)
        
        # Enhanced interpretation
        confidence = analysis['confidence']
        variant = analysis['variant']
        type_info = PERSONALITY_TYPES[analysis['type']]
        
        if confidence >= 0.4:
            interpretation = f"🎯 Strong {analysis['type']} Profile: Your personality type is clearly defined with high confidence. You consistently demonstrate the core traits of {type_info['name']}, showing authentic alignment with this type's natural patterns of thinking, feeling, and behaving."
        elif confidence >= 0.2:
            interpretation = f"⚖️ Moderate {analysis['type']} Profile: You show clear tendencies toward {type_info['name']} traits while maintaining flexibility across different situations. This suggests adaptability and situational awareness in how you express your personality."
        else:
            interpretation = f"🌈 Balanced {analysis['type']} Profile: You demonstrate a nuanced blend of personality traits centered around the {type_info['name']} type. This versatility allows you to adapt your approach based on context and circumstances."
        
        if variant == "A":
            interpretation += " Your Assertive variant suggests confidence in your abilities and decisions, with natural resilience to stress."
        else:
            interpretation += " Your Turbulent variant indicates thoughtful self-reflection and a drive for continuous improvement."
        
        st.markdown(f"""
        <div class="interpretation-box">
            <h3>💡 Advanced Interpretation</h3>
            <p>{interpretation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Download enhanced results
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retake Assessment", key="retake_btn"):
                # Reset everything for new assessment
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col2:
            results_text = f"""Advanced Adaptive Personality Assessment Results
=============================================

Personality Type: {analysis['full_type']} - {type_info['name']}
Scenario: {scenario['name']}
Confidence Level: {confidence:.1%}
Questions Asked: {len(st.session_state.questions_asked)} adaptive questions

CORE DESCRIPTION:
{type_info['description']}

STRENGTHS:
{type_info['strengths']}

GROWTH AREAS:
{type_info['weaknesses']}

CAREER MATCHES:
{type_info['careers']}

MBTI DIMENSION SCORES:
Extraversion/Introversion: {analysis['mbti_scores']['E/I']:.1f}/10
Sensing/Intuition: {analysis['mbti_scores']['S/N']:.1f}/10  
Thinking/Feeling: {analysis['mbti_scores']['T/F']:.1f}/10
Judging/Perceiving: {analysis['mbti_scores']['J/P']:.1f}/10

ADVANCED TRAIT ANALYSIS:
Leadership: {analysis['traits']['leadership']:.1f}/10
Creativity: {analysis['traits']['creativity']:.1f}/10
Empathy: {analysis['traits']['empathy']:.1f}/10
Risk Tolerance: {analysis['traits']['risk_tolerance']:.1f}/10
Perfectionism: {analysis['traits']['perfectionism']:.1f}/10
Social Harmony: {analysis['traits']['social_harmony']:.1f}/10

INTERPRETATION:
{interpretation}
            """
            st.download_button(
                label="📄 Download Complete Analysis",
                data=results_text,
                file_name=f"adaptive_personality_analysis_{analysis['full_type']}.txt",
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
