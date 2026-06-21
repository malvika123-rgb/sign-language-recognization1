import tkinter as tk
from tkinter import messagebox
import json
import subprocess
import os

# User database file
USER_FILE = "users.json"

# Load users
if os.path.exists(USER_FILE):
    with open(USER_FILE, "r") as f:
        users = json.load(f)
else:
    users = {}

# Login function
def login():
    username = entry_username.get()
    password = entry_password.get()
    if username in users and users[username] == password:
        messagebox.showinfo("Login Success", f"Welcome, {username}!")
        root.destroy()  # Close login GUI

        # Launch gesture recognition script
        subprocess.Popen(["python", "inference_classifier.py"])
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

# Register function
def register():
    username = entry_username.get()
    password = entry_password.get()
    if not username or not password:
        messagebox.showerror("Error", "Please enter username and password")
        return
    if username in users:
        messagebox.showerror("Error", "Username already exists")
        return
    users[username] = password
    with open(USER_FILE, "w") as f:
        json.dump(users, f)
    messagebox.showinfo("Success", f"User '{username}' registered!")

# GUI
root = tk.Tk()
root.title("Sign Language Project - Login")
root.geometry("400x250")

tk.Label(root, text="Username:").pack(pady=10)
entry_username = tk.Entry(root)
entry_username.pack()

tk.Label(root, text="Password:").pack(pady=10)
entry_password = tk.Entry(root, show="*")
entry_password.pack()

tk.Button(root, text="Login", command=login).pack(pady=10)
tk.Button(root, text="Register", command=register).pack(pady=5)

root.mainloop()
