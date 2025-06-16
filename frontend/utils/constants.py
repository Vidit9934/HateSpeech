API_BASE_URL = "http://localhost:8000"  # Base URL for all endpoints

EXAMPLE_TEXTS = {
    "Neutral": "Hello everyone, hope you're having a great day!",
    "Offensive": "You're such an idiot, learn to write properly!",
    "Toxic": "I'll make sure you never work in this industry again.",
    "Hate": "All [group] should be banned from our country.",
}

ACTION_COLORS = {
    "ban": "error",
    "warn": "warning",
    "flag": "warning",
    "allow": "success",
    "review": "info"
}

# Add Reddit-specific constants
EXAMPLE_REDDIT_URLS = [
    "https://www.reddit.com/r/Python/comments/example1",
    "https://www.reddit.com/r/learnprogramming/comments/example2"
]