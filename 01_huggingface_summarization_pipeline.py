from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization")

text = """
Artificial Intelligence is transforming industries across the world.

In healthcare, AI assists doctors in diagnosing diseases earlier.
In finance, AI helps detect fraud and automate complex trading systems.
In education, AI personalizes learning experiences for students.
In transportation, autonomous vehicles use AI to navigate roads.
Governments and companies are investing billions into AI research.
However, concerns about ethics, bias, privacy, and job displacement continue to grow.
Experts believe AI will reshape the global economy over the next decade.
"""

summary = summarizer(text, max_length=120, min_length=20, do_sample=False)

print(summary[0]['summary_text'])
