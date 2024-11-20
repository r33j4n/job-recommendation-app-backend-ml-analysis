import re

def extract_numeric_experience(experience_str):
    numbers = re.findall(r'\d+(?:\.\d+)?', experience_str)
    if numbers:
        return float(numbers[0])
    else:
        return 0.0  # Default to 0 if no numeric value is found

def parse_skills(skills_str):
    return set(skill.strip().lower() for skill in skills_str.split(',') if skill.strip())