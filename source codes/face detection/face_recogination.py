import cv2
import face_recognition
import pickle
import csv
import datetime
import numpy as np
import tkinter as tk
from tkinter import Label, Button, ttk
import threading
import os
from PIL import Image, ImageTk


# Function to mark attendance in CSV and automatically mark absentees
def mark_attendance(name):
    filename = "attendance.csv"
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    file_exists = os.path.isfile(filename)

    # Read existing data
    data = {}
    header = ["Name"]

    if file_exists:
        with open(filename, "r") as file:
            reader = csv.reader(file)
            rows = list(reader)
            if rows:
                header = rows[0]  # First row contains column names
                for row in rows[1:]:
                    data[row[0]] = row[1:]

    # Ensure today's date is in the header
    if today_date not in header:
        header.append(today_date)
        for key in data.keys():
            data[key].append("Absent")  # Default all to absent for new date

    # Mark attendance only if not already marked for today
    if name in data:
        index = header.index(today_date) - 1
        if data[name][index] != "Present":
            data[name][index] = "Present"
    else:
        data[name] = ["Absent"] * (len(header) - 1)
        data[name][-1] = "Present"

    # Write back to CSV
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for key, values in data.items():
            writer.writerow([key] + values)

    print(f"Attendance marked for {name} on {today_date}")


# Load the trained face encodings and names
def load_trained_faces():
    with open('faces.pkl', 'rb') as f:
        return pickle.load(f)


# GUI Application Class
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("900x700")
        self.root.configure(bg='black')

        # Setup styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        self.style.configure('TButton', font=('Helvetica', 10), padding=5)
        self.style.configure('Start.TButton', foreground='white', background='#2ecc71')
        self.style.configure('Stop.TButton', foreground='white', background='#e74c3c')
        self.style.configure('Show.TButton', foreground='white', background='#3498db')
        
        # Create canvas for animated border
        self.canvas = tk.Canvas(root, width=640, height=480, bg='black', highlightthickness=0)
        self.canvas.pack(pady=10)

        # Label to show camera feed
        self.label = Label(self.canvas, bd=0, bg='black')
        self.label.place(x=5, y=5)

        # Create animated border
        self.border_id = self.canvas.create_rectangle(5, 5, 635, 475, outline='#2ecc71', width=3)

        # Feedback label
        self.feedback_label = Label(root, text="", font=('Helvetica', 12), bg='black', fg='white')
        self.feedback_label.pack(pady=5)

        # Buttons frame
        button_frame = tk.Frame(root, bg='black')
        button_frame.pack(pady=10)

        self.start_button = ttk.Button(button_frame, text="Start Recognition", 
                                     style='Start.TButton', command=self.start_recognition)
        self.start_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Recognition", 
                                    style='Stop.TButton', command=self.stop_recognition)
        self.stop_button.pack(side='left', padx=5)
        
        self.show_attendance_button = ttk.Button(button_frame, text="Show Attendance", 
                                               style='Show.TButton', command=self.load_attendance)
        self.show_attendance_button.pack(side='left', padx=5)

        # Add hover effects
        self.add_button_hover_effect(self.start_button, '#27ae60', '#2ecc71')
        self.add_button_hover_effect(self.stop_button, '#c0392b', '#e74c3c')
        self.add_button_hover_effect(self.show_attendance_button, '#2980b9', '#3498db')

        # Table to show attendance (Initially hidden)
        self.tree = ttk.Treeview(root)
        
        # Configure treeview style
        self.style.configure('Custom.Treeview', 
                           font=('Helvetica', 10),
                           rowheight=25,
                           background='#f0f0f0',
                           fieldbackground='#e0e0e0',
                           foreground='black')
        self.style.configure('Custom.Treeview.Heading', 
                           font=('Helvetica', 11, 'bold'),
                           background='#808080',
                           foreground='white')
        self.style.map('Custom.Treeview', 
                      background=[('selected', '#007bff')])

        # Load trained faces
        self.known_face_encodings, self.known_face_names = load_trained_faces()

        # Initialize video capture
        self.video_capture = cv2.VideoCapture(0)
        self.running = False
        self.pulse_active = False
        self.update_frame()

    def add_button_hover_effect(self, button, hover_color, normal_color):
        button.bind("<Enter>", lambda e: button.config(style=f'{button["style"]}.Hover'))
        button.bind("<Leave>", lambda e: button.config(style=button["style"].replace('.Hover','')))
        self.style.configure(f'{button["style"]}.Hover', background=hover_color)

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(img)
            self.label.config(image=img)
            self.label.image = img

        if self.running:
            self.root.after(10, self.update_frame)

    def start_recognition(self):
        if not self.running:
            self.running = True
            self.pulse_active = True
            self.start_pulse_animation()
            threading.Thread(target=self.recognize_faces, daemon=True).start()

    def stop_recognition(self):
        self.running = False
        self.pulse_active = False

    def start_pulse_animation(self):
        if self.pulse_active:
            self.pulse_border()
            self.root.after(100, self.start_pulse_animation)

    def pulse_border(self):
        current_color = self.canvas.itemcget(self.border_id, "outline")
        colors = ['#27ae60', '#2ecc71', '#27ae60', '#1abc9c']
        next_color = colors[(colors.index(current_color)+1) % len(colors)] if current_color in colors else colors[0]
        self.canvas.itemconfig(self.border_id, outline=next_color)

    def show_feedback(self, name):
        self.feedback_label.config(text=f"Attendance Marked: {name}", fg='white', bg='#2ecc71')
        self.feedback_label.place(x=400, y=500, anchor='center')
        self.root.after(1500, self.fade_feedback)

    def fade_feedback(self):
        for alpha in range(100, -10, -10):
            self.feedback_label.config(fg=f'#{alpha:02x}{alpha:02x}{alpha:02x}')
            self.root.update()
            self.root.after(50)
        self.feedback_label.config(text="", bg='black')

    def recognize_faces(self):
        while self.running:
            ret, frame = self.video_capture.read()
            if not ret:
                continue

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                    mark_attendance(name)
                    self.root.after(0, self.show_feedback, name)

                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Convert frame to Tkinter format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(img)
            self.label.config(image=img)
            self.label.image = img

    def load_attendance(self):
        """Loads attendance from CSV and displays it in a table"""
        filename = "attendance.csv"

        if not os.path.exists(filename):
            return

        self.tree.pack_forget()
        self.tree = ttk.Treeview(self.root, style='Custom.Treeview')
        
        self.tree.pack(pady=10, fill='both', expand=True)

        with open(filename, "r") as file:
            reader = csv.reader(file)
            rows = list(reader)

            if not rows:
                return

            self.tree["columns"] = rows[0]
            self.tree["show"] = "headings"

            for col in rows[0]:
                self.tree.heading(col, text=col, anchor='center')
                self.tree.column(col, width=100, anchor='center')

            for row in rows[1:]:
                self.tree.insert("", "end", values=row)

    def __del__(self):
        self.video_capture.release()
        cv2.destroyAllWindows()


# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()