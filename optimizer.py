import dspy
from dspy.teleprompt import BootstrapFewShot

# 1. Connect to local LM Studio server
lm = dspy.LM(
    model="openai/custom-model", 
    api_base="http://localhost:1234/v1", 
    api_key="lm-studio"
)
dspy.configure(lm=lm)

# 2. Define the Target Signature
class MedicalRouter(dspy.Signature):
    """Route emergency medical descriptions to the correct hospital specialty."""
    
    symptoms: str = dspy.InputField(desc="The reported patient symptoms or incident description")
    department: str = dspy.OutputField(desc="The hospital department target (e.g., Cardiology, Neurology, Pediatrics)")

# 3. Build a Small Evaluation/Training Dataset
train_data = [
    dspy.Example(symptoms="Chest pains radiating down left arm, sweating.", department="Cardiology").with_inputs("symptoms"),
    dspy.Example(symptoms="5-year-old with sudden high fever and ear rash.", department="Pediatrics").with_inputs("symptoms"),
    dspy.Example(symptoms="Slurred speech, facial droop on right side.", department="Neurology").with_inputs("symptoms"),
    dspy.Example(symptoms="Severe palpitations after drinking three energy drinks.", department="Cardiology").with_inputs("symptoms"),
    dspy.Example(symptoms="Toddler swallowed a small plastic toy part.", department="Pediatrics").with_inputs("symptoms")
]

# 4. Define an Automated Success Metric Function
def accuracy_metric(gold_data, model_prediction, trace=None):
    return gold_data.department.strip().lower() == model_prediction.department.strip().lower()

# 5. Initialize and Run the Algorithmic Compiler
optimizer = BootstrapFewShot(metric=accuracy_metric, max_bootstrapped_demos=2)
base_program = dspy.ChainOfThought(MedicalRouter)

print("Starting local prompt compilation against training dataset...")
optimized_program = optimizer.compile(base_program, trainset=train_data)
print("Compilation complete! Best instructions synthesized.")

# 6. Test your newly optimized local asset
final_test = optimized_program(symptoms="8-year-old cannot stop coughing and wheezing.")
print(f"\nFinal Test Results:")
print(f"Target Department: {final_test.department}")