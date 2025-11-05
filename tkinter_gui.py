
import tkinter as tk
from ensemble import ensemble
from normalize import normalize_func


def generate():
    input_values = [float(entry.get()) for entry in entry_widgets]

    normalized_values = normalize_func(input_values)

    pred_class,pred_value = ensemble(normalized_values)
    sigma=617.78
    mu=589.08 #values taken from unnormalized website, https://archive.ics.uci.edu/dataset/211/communities+and+crime+unnormalized where it has summary stats for each variable
    pred_value=pred_value * (6 * sigma) + (mu - 3 * sigma)

    if pred_class==0:
        class_name='low crime'
    else:
        class_name='high crime'

    if pred_value<0:
        pred_value=0 #prevents prediction of negative crime rate

    result_text = (
        f"This neighborhood is predicted to be a {class_name} community,\n"
        f"with {round(pred_value[0], 2)} violent crimes per 100k people."
    )
    result_label.config(text=result_text)


# Set up the main window
root = tk.Tk()
root.title("Input Generator")
root.geometry("300x400")

instruction_label = tk.Label(root, text="Enter the percentage of people that are ...", font=("Helvetica", 10, "italic"))
instruction_label.pack(pady=(10, 10))


# Input field labels
labels = [
    'African-American', 'Caucasian', 'Have Investment or Rent Income','Have Public Assistance Income',
    'Under the Poverty Line', 'Divorced', 'Kids Living in a 2-Parent Household',
    'Living in Owner-Occupied Housing'
]

# Store Entry widgets
entry_widgets = []

for i, label_text in enumerate(labels):
    label = tk.Label(root, text=label_text)
    label.pack(pady=(10 if i == 0 else 5, 2))

    entry = tk.Entry(root, width=30)
    entry.pack(pady=2)
    entry_widgets.append(entry)

# Generate button
generate_button = tk.Button(root, text="Predict", command=generate)
generate_button.pack(pady=20)

result_label = tk.Label(root, text="", wraplength=280, justify="left")
result_label.pack(pady=10)

# Start the main event loop
root.mainloop()
