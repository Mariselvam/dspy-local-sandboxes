import dspy

# 1. Connect to LM Studio's local endpoint
lm = dspy.LM(
    # Force 'openai/' prefix before your local model's name
    model="openai/qwen3.5-9b", 
    api_base="http://localhost:1234/v1", 
    api_key="lm-studio" # Keep the dummy key string
)
dspy.configure(lm=lm)

# 2. Define your Data Contract (Signature)
class SentimentClassifier(dspy.Signature):
    """Classify the sentiment of corporate feedback text."""
    
    feedback_text: str = dspy.InputField(desc="A customer review or corporate feedback string")
    sentiment: str = dspy.OutputField(desc="Must be exactly one of: Positive, Negative, or Neutral")

# 3. Pass the contract to a Reasoning Module
analyzer = dspy.ChainOfThought(SentimentClassifier)

# 4. Execute locally
result = analyzer(feedback_text="The setup took 2 hours and documentation was missing, but customer support fixed it fast.")

print(f"Reasoning: {result.reasoning}")
print(f"Final Decision: {result.sentiment}")