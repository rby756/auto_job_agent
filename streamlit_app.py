#!/usr/bin/env python3
"""
Streamlit UI for Auto Job Agent
"""
import os
import sys
import requests
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configuration
API_BASE_URL = "http://localhost:8000"  # Update this if your API is running elsewhere

# Set page config
st.set_page_config(
    page_title="Auto Job Agent",
    page_icon="💼",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        max-width: 1000px;
        padding: 2rem;
    }
    .job-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .job-score {
        font-size: 0.9em;
        color: #666;
        margin-bottom: 0.5rem;
    }
    .job-title {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .job-company {
        color: #2c7be5;
        margin-bottom: 0.5rem;
    }
    .job-description {
        line-height: 1.5;
    }
    .stProgress > div > div > div > div {
        background-color: #2c7be5;
    }
</style>
""", unsafe_allow_html=True)

def check_api_connection() -> bool:
    """Check if the API is running and accessible."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def process_resume(file) -> str:
    """Process resume file using the OCR API."""
    try:
        with st.spinner("Processing resume..."):
            response = requests.post(
                f"{API_BASE_URL}/api/ocr/process",
                files={"file": file},
                params={"lang": "eng"}
            )
            response.raise_for_status()
            result = response.json()
            if result.get("success"):
                return result["text"]
            else:
                st.error(f"Error processing resume: {result.get('error', 'Unknown error')}")
                return ""
    except Exception as e:
        st.error(f"Failed to process resume: {str(e)}")
        return ""

def get_job_matches(resume_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Get job matches for the resume text."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/jobs/match",
            json={
                "resume_text": resume_text,
                "top_k": top_k,
                "score_threshold": 0.3  # Lower threshold to get more results
            }
        )
        response.raise_for_status()
        result = response.json()
        return result.get("matches", [])
    except Exception as e:
        st.error(f"Failed to get job matches: {str(e)}")
        return []

def display_job_card(job: Dict[str, Any], index: int):
    """Display a job card with match score and details."""
    with st.container():
        # Create columns for score bar and job details
        col1, col2 = st.columns([1, 10])
        
        with col1:
            # Display score as a progress bar
            score = job.get("score", 0)
            st.metric("Match", f"{score*100:.0f}%")
            
        with col2:
            # Display job title and company
            metadata = job.get("metadata", {})
            source = job.get("source", "")
            title = metadata.get("title") or Path(source).stem if source else "Job Title Not Available"
            st.markdown(f"<div class='job-title'>{title}</div>", 
                       unsafe_allow_html=True)
            
            if "company" in metadata:
                st.markdown(f"<div class='job-company'>{metadata['company']}</div>", 
                           unsafe_allow_html=True)
            
            # Display job description with a "Read more" expander
            with st.expander("View Job Details"):
                st.markdown(f"<div class='job-description'>{job.get('text', '')}</div>", 
                           unsafe_allow_html=True)
                
                # Display additional metadata if available
                if "location" in metadata:
                    st.caption(f"📍 {metadata['location']}")
                if "salary" in metadata:
                    st.caption(f"💰 {metadata['salary']}")
                
                # Add apply button
                if "apply_url" in metadata:
                    st.link_button("Apply Now", metadata["apply_url"])
        
        st.divider()

def main():
    """Main Streamlit app."""
    st.title("💼 Auto Job Agent")
    st.markdown("Upload your resume and discover matching job opportunities!")
    
    # Check API connection
    if not check_api_connection():
        st.error("⚠️ Could not connect to the API. Please make sure the API server is running.")
        st.info("Start the API server with: `uvicorn api_app.main:app --reload`")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF, JPG, or PNG)", 
        type=["pdf", "jpg", "jpeg", "png"]
    )
    
    # Process resume and show matches when a file is uploaded
    if uploaded_file is not None:
        # Display the uploaded resume
        st.subheader("Your Resume")
        
        # Show file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.1f} KB",
            "File type": uploaded_file.type
        }
        st.json(file_details)
        
        # Process the resume
        resume_text = process_resume(uploaded_file)
        
        if resume_text:
            # Show extracted text in an expander
            with st.expander("View Extracted Text"):
                st.text_area("Extracted Text", resume_text, height=200)
            
            # Get job matches
            st.subheader("🔍 Recommended Jobs")
            st.markdown("Here are the jobs that best match your resume:")
            
            with st.spinner("Finding matching jobs..."):
                matches = get_job_matches(resume_text, top_k=5)
                
                if matches:
                    # Sort by score in descending order
                    sorted_matches = sorted(matches, key=lambda x: x.get("score", 0), reverse=True)
                    
                    # Display job cards
                    for i, job in enumerate(sorted_matches, 1):
                        display_job_card(job, i)
                else:
                    st.warning("No matching jobs found. Try adjusting your resume or check back later.")
                    
                    # Show sample resume tips
                    with st.expander("Tips to improve your resume"):
                        st.markdown("""
                        - Include relevant skills and technologies
                        - Add quantifiable achievements
                        - Use industry-standard keywords
                        - Keep the format clean and professional
                        - Highlight relevant experience
                        """)
        
        # Add a refresh button
        if st.button("🔄 Find More Jobs"):
            st.rerun()
    
    # Add a footer
    st.markdown("---")
    st.caption("Auto Job Agent - Find your next career opportunity")

if __name__ == "__main__":
    main()
