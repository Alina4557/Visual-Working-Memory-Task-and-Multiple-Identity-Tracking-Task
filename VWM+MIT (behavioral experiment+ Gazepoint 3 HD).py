# -*- coding: utf-8 -*-
from psychopy import visual, core, event, data, gui, monitors
import random
import pandas as pd
import math
import sys
import os
import numpy as np
import threading
import socket
import json
import matplotlib.pyplot as plt
import traceback
import re

# Convert eye-tracking data into table
def parse_rec_xml(xml_str):
    """
    <REC FPOGX="1.03000" FPOGY="0.87743" FPOGS="649.40576" FPOGD="0.27448" 
         FPOGID="458" FPOGV="1" BKID="0" BKDUR="0.00000" BKPMIN="21" />
    """
    pattern = r'(\w+)="([^"]*)"'
    matches = re.findall(pattern, xml_str)
    data = {"type": "REC"}  # Add type fiel
    for key, value in matches:
        try:
            data[key] = float(value)
        except ValueError:
            data[key] = value
    return data

# Set stdout encoding
sys.stdout.reconfigure(encoding='utf-8')

# Gazepoint Socket client class definition
class GazePointSocketClient:
    def __init__(self, ip="10.56.209.106", port=4242):
        self.ip = ip
        self.port = port
        self.socket = None
        self.running = False
        self.data_buffer = []

    def connect(self):
        """Connect to Gazepoint data stream server and enable required data types"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.ip, self.port))
            # Send commands to enable eye-tracking data
            # Gazepoint Data Channel Extension Configuration (Get All Real-Time Fields)
            self.socket.sendall(b'<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n')             # Enable core data
            self.socket.sendall(b'<SET ID="ENABLE_SEND_POG_FIX" STATE="1" />\r\n')         # Fixation points
            self.socket.sendall(b'<SET ID="ENABLE_SEND_BLINK" STATE="1" />\r\n')           # Blink events
            self.socket.sendall(b'<SET ID="ENABLE_SEND_SACCADE" STATE="1" />\r\n')         # Saccades
            self.socket.sendall(b'<SET ID="ENABLE_SEND_EYE_LEFT" STATE="1" />\r\n')        # Left eye LPCX, LPMM
            self.socket.sendall(b'<SET ID="ENABLE_SEND_EYE_RIGHT" STATE="1" />\r\n')       # Right eye RPCX, RPMM
            self.socket.sendall(b'<SET ID="ENABLE_SEND_PUPILMM" STATE="1" />\r\n')         # Pupil diameter only (for legacy devices)
            self.socket.sendall(b'<SET ID="ENABLE_SEND_TIME" STATE="1" />\r\n')            # Timestamps
            return True
        except Exception as e:
            print(f"[Connection failed] Unable to connect to Gazepoint: {e}")
            return False

    def start_recording(self):
        """Start background thread to record gaze data"""
        if not self.socket:
            print("Please connect the socket first")
            return
        self.running = True
        self.data_buffer = []
        # Start new thread to continuously receive data
        self.thread = threading.Thread(target=self._receive_data, daemon=True)
        self.thread.start()

    def stop_recording(self):
        """Stop recording gaze data"""
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)

    def get_events(self):
        """et all gaze event data currently in the buffer"""
        return self.data_buffer.copy()

    def _receive_data(self):
        cnt = 0 
        print("📡 Gaze data thread starte")
        while self.running:
            try:
                data_bytes = self.socket.recv(4096)
                if not data_bytes:
                    print("❌ GGazepoint data stream interrupted (empty response)")
                    break
                data_str = data_bytes.decode("utf-8")
    
                # ✅  Support parsing multiple <REC ... /> at once
                for rec in data_str.strip().split("\r\n"):
                    if rec.startswith("<REC") and rec.endswith("/>"):
                        try:
                            data_dict = parse_rec_xml(rec)
                            data_dict["CNT"] = cnt
                            self.data_buffer.append(data_dict)
                            cnt += 1
                            if cnt % 100 == 0:
                                print(f"📥 gaze 已收 {cnt} 条")
                        except Exception as e:
                            print("[Parsing error]", e)
    
            except socket.timeout:
                pass  # ✅ Ignore timeout to avoid spamming
            except Exception as e:
                print("[Receiving error]", e)
                break
    
# Eye-tracking data handler class definition
class EyeTracking:
    def __init__(self):
        # Initialize and connect to Gazepoint
        self.event_log = [] 
        self.tracker = GazePointSocketClient(ip="10.56.209.106", port=4242)
        if not self.tracker.connect():
            raise RuntimeError("Unable to connect to Gazepoint eye tracker. Please ensure Gazepoint Control is running.")
        self.tracker.socket.settimeout(0.2)
        # Print successful connection message
        print("Gazepoint connected successfully!")
        # end commands to enable data stream to ensure all necessary gaze data is received
        self.tracker.socket.sendall(b'<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n')
        self.tracker.socket.sendall(b'<SET ID="ENABLE_SEND_POG_FIX" STATE="1" />\r\n')
        self.tracker.socket.sendall(b'<SET ID="ENABLE_SEND_BLINK" STATE="1" />\r\n')
        self.tracker.socket.sendall(b'<SET ID="ENABLE_SEND_SACCADE" STATE="1" />\r\n')

    def export_gaze_events_to_csv(self, filename="gaze_events.csv"):
        all_events = []
        for trial_index, trial in enumerate(data_list):
            for phase_label in ["VWM_stim_events", "VWM_test_events", "MIT_stim_events", "MIT_track_events", "MIT_test_events"]:
                if phase_label in trial:
                    events = trial[phase_label]
                    for e in events:
                        e_copy = e.copy()
                        e_copy["trial_index"] = trial_index + 1
                        e_copy["task"] = trial.get("task", "")
                        e_copy["phase"] = trial.get("phase", "")
                        e_copy["with_instruction"] = trial.get("with_instruction", "")
                        e_copy["event_phase"] = phase_label.replace("_events", "")
                        if trial["task"] in ["VWM", "VWM+MIT"]:
                            e_copy["VWM_condition"] = trial.get("VWM_condition", "")
                            e_copy["VWM_correct"] = trial.get("VWM_correct", "")
                        if trial["task"] in ["MIT", "VWM+MIT"]:
                            e_copy["MIT_size"] = trial.get("MIT_size", "")
                            e_copy["MIT_correct"] = trial.get("MIT_correct", "")
                        all_events.append(e_copy)
    
        if not all_events:
            print("⚠️ No gaze data was recorded. Skipping CSV export")
            return ""
    
        events_df = pd.DataFrame(all_events)
        priority_cols = [
            "trial_index", "task", "phase", "with_instruction", "event_phase",
            "VWM_condition", "VWM_correct", "MIT_size", "MIT_correct"
        ]
        other_cols = [col for col in events_df.columns if col not in priority_cols]
        events_df = events_df[priority_cols + other_cols]
        events_df.to_csv(filename, index=False, encoding="utf-8")
        return filename

    def visualize_gaze_events(self, gaze_csv_path="gaze_events.csv", output_dir="visualizations", max_trials=10, screen_size=(1280,1024)):
        """Read gaze event CSV and plot scanpath for each trial"""
        try:
            df = pd.read_csv(gaze_csv_path)
        except Exception as e:
            print(f"[Visualization Error] Unable to read CSV: {e}")
            return
        os.makedirs(output_dir, exist_ok=True)
        # Only draw the first 'max_trials' trials
        for trial_idx in sorted(df["trial_index"].unique())[:max_trials]:
            trial_df = df[df["trial_index"] == trial_idx]
            # If coordinates are normalized (0~1), convert them to pixel coordinates (assume screen 1280x1024)
            if trial_df["gaze_x"].max() <= 1.0 and trial_df["gaze_y"].max() <= 1.0:
                trial_df["gaze_x"] *= screen_size[0]
                trial_df["gaze_y"] *= screen_size[1]
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.set_title(f"Trial {trial_idx} Gaze Scanpath")
            ax.set_xlim(0, win.size[0]); ax.set_ylim(0, win.size[1])
            ax.invert_yaxis()  # Invert Y-axis to make (0,0) the top-left corner
            ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
            # Draw trajectory lines
            ax.plot(trial_df["gaze_x"], trial_df["gaze_y"], linestyle="-", alpha=0.5)
            # Label each fixation point with index
            for i, (x, y) in enumerate(zip(trial_df["gaze_x"], trial_df["gaze_y"])):
                ax.text(x, y, str(i), fontsize=6, alpha=0.6)
            output_file = os.path.join(output_dir, f"trial_{trial_idx:03d}_scanpath.png")
            plt.savefig(output_file)
            plt.close()

# Instruction class (based on original script)
class Instructions:
    def __init__(self, win):
        self.win = win
        self.mouse = event.Mouse()
    def wait_for_mouse(self, timeout=None):
        timer = core.Clock()
        while any(self.mouse.getPressed()):
            core.wait(0.05)
        while True:
            if any(self.mouse.getPressed()):
                break
            elif "escape" in event.getKeys():
                save_data_and_exit()
            elif timeout and timer.getTime() >= timeout:
                break
            core.wait(0.1)
    def begin_instructions(self):
        instructions = visual.TextStim(
            self.win,
            text=(
                "Welcome to the experiment!\n\n"
                "The experiment will take around 30 minutes in total.\n"
                "It includes three different tasks.\n"
                "Before the formal experiment begins, you will first complete some practice trials "
                "until you fully understand each task.\n\n"
                "You’ll get feedback in practice, but not in the formal task.\n"
                "Click the left or right mouse button to start the practice."
            ),
            color="black", height=0.7, wrapWidth=25, pos=(0, 0)
        )
        instructions.draw()
        self.win.flip()
        while True:
            if any(self.mouse.getPressed()):
                break
            elif "escape" in event.getKeys():
                save_data_and_exit()
            core.wait(0.1)
    def VWM_show_instructions(self):
        instructions = visual.TextStim(self.win, text="Task 1\n\nYou need to remember the color and the location of three circles\nClick Mouse button (left or right) to start.",
                                       color="black", height=0.7, wrapWidth=25, pos=(0, 0))
        instructions.draw()
        self.win.flip()
        self.wait_for_mouse()
    def MIT_show_instructions(self):
        instruction_text = visual.TextStim(
            self.win,
            text=(
                "Task 2\n\n"
                "The objects with borders (targets) are the ones you need to track.\n\n"
                "Once the borders disappear, everything will start moving — just focus on your targets and remember where they go.\n\n"
                "When the motion stops, one object will be highlighted. Click the highlighted object if it was a target, or click NO if it was not."
            ),
            color="black", height=0.7, wrapWidth=25, pos=(0, 0)
        )
        instruction_text.draw()
        self.win.flip()
        self.wait_for_mouse()
    def VWM_MIT_show_instructions(self):
        instruction_text = visual.TextStim(
            self.win,
            text=(
                "Task 3\n\n"
                "This is a combined task of 1 and 2.\n"
                "You need to do Task 1 and Task 2 at the same time.\n\n"
                "Remember the three colored circles and track the bordered targets.\n"
                "You will answer Task 1 first, then Task 2.\n\n"
                "Click to start practice of Task 3."
            ),
            color="black", height=0.7, wrapWidth=25, pos=(0, 0)
        )
        instruction_text.draw()
        self.win.flip()
        self.wait_for_mouse()
    def formal_experiment_instructions(self):
        instruction_text = visual.TextStim(
            self.win,
            text=(
                "The practice session is over. Now the formal experiment will begin.\n"
                "Please focus and do your best. There will be a short break halfway through.\n\n"
                "Click to start the formal experiment."
            ),
            color="black", height=0.7, wrapWidth=25, pos=(0, 0)
        )
        instruction_text.draw()
        self.win.flip()
        self.wait_for_mouse()
    def break_instructions(self):
        instructions = visual.TextStim(
            self.win,
            text="You can take a short break now.\n\nClick to continue the experiment when ready.",
            color="black", height=0.7, wrapWidth=25, pos=(0, 0)
        )
        instructions.draw()
        self.win.flip()
        while True:
            if any(self.mouse.getPressed()):
                break
            elif "escape" in event.getKeys():
                save_data_and_exit()
            core.wait(0.1)
    def end_instructions(self):
        thank_you_text = visual.TextStim(
            self.win,
            text="Thank you for your participation!\n\nThe experiment will close automatically in 2 seconds.",
            color="black", height=0.7, wrapWidth=25, pos=(0, 0)
        )
        thank_you_text.draw()
        self.win.flip()
        core.wait(2.0)

# Feedback function (used in practice to indicate correctness)
def show_feedback(vwm_correct=None, mit_correct=None):
    feedback_text = ""
    if vwm_correct is not None:
        feedback_text += f"Task1: {'Correct ✅' if vwm_correct else 'Wrong ❌'}\n"
    if mit_correct is not None:
        feedback_text += f"Task2: {'Correct ✅' if mit_correct else 'Wrong ❌'}"
    feedback = visual.TextStim(win, text=feedback_text, color="black", height=0.8, wrapWidth=20)
    feedback.draw()
    win.flip()
    core.wait(1.5)

# Initialize experiment window and parameters
exp_info = {'Participant': ''}
dlg = gui.DlgFromDict(dictionary=exp_info, title='VWM Experiment')
if not dlg.OK:
    core.quit()

monitor_name = "testMonitor"
my_monitor = monitors.Monitor(monitor_name)
my_monitor.setSizePix([1920,1080])
my_monitor.setWidth(31)   # Screen width (cm)
my_monitor.setDistance(57)
win = visual.Window(
    size=[1920,1080],
    monitor=my_monitor,
    color="white",
    units="deg",
    fullscr=True
)

# Unit conversion: convert deg to normalized coordinates
screen_width_deg = 31   # Approximate horizontal visual angle width
screen_height_deg = 17.42  # Approximate vertical visual angle height
def convert_normalized_to_deg(gx, gy):
    converted_x = (gx - 0.5) * screen_width_deg
    # If PsychoPy uses Y up, (0.5, 0.5) corresponds to y=0:
    converted_y = (0.5 - gy) * screen_height_deg
    return converted_x, converted_y

# Calculate distance in degrees from center to screen edge
screen_width_deg = 31  # Approximately corresponds to 31° of visual angle (based on monitor specs)
stimulus_radius = (screen_width_deg / 4) - 1  # Set the radius to 1/4 of the screen width, slightly inward

# Define fixation point and retention interval prompt
def Fixation():
    fixation = visual.TextStim(win, text="+", color="black", height=1, pos=(0, 0))
    fixation.draw()
    win.flip()
    core.wait(0.5)

def Retention_Interval():
    fixation = visual.TextStim(win, text="+", color="black", height=1, pos=(0, 0))
    fixation.draw()
    win.flip()
    core.wait(10.2)#10.2

# Define task type prompt display
def Trial_begin(label, duration=2.0):
    number = visual.TextStim(win, text=label, color="red", height=1, pos=(0, 0))
    number.draw()
    win.flip()
    core.wait(duration)

# Data saving on abnormal exit
def save_data_and_exit():
    try:
        tracker.stop_recording()
        gaze_events = tracker.get_events()
        named_gaze_events = [{f"ESC_exit_{k}": v for k, v in e.items()} for e in gaze_events]

        # ✅ Add event log to support export
        if hasattr(eye_tool, "event_log"):
            eye_tool.event_log.append(named_gaze_events)
        else:
            print("⚠️ eye_tool has no event_log, cannot export gaze")

    except Exception as e:
        print("[Error stopping recording on exit]", e)

    os.makedirs("data", exist_ok=True)
    filename = f"data/{exp_info['Participant']}_partial"
    df = pd.DataFrame(data_list)
    df.to_excel(filename + ".xlsx", index=False)
    df.to_csv(filename + ".csv", index=False)

    # ✅ Proactively call gaze data export
    try:
        gaze_csv = eye_tool.export_gaze_events_to_csv(f"{filename}_gaze_events.csv")
        pd.read_csv(gaze_csv).to_excel(f"{filename}_gaze_events.xlsx", index=False)
        print("✅ gaze data exported")
    except Exception as e:
        print(f"⚠️ gaze data export failed: {e}")

    print("🚨 Experiment aborted, current data saved")
    win.close()
    core.quit()

# Dialog to confirm task understanding
def confirm_understanding():
    mouse = event.Mouse()
    rt_clock = core.Clock()
    instruction = (
        "Do you understand this task procedure?\n\n"
        "YES: Click Left mouse button,\n"
        "You will proceed to practicing this task without instructions,\n"
        "but still need to respond to the task.\n\n"
        "NO: Click Right mouse button,\n"
        "You will repeat this practice with instruction.\n\n"
    )
    text_stim = visual.TextStim(win, text=instruction, color='black', height=0.7, wrapWidth=25)
    text_stim.draw()
    win.flip()
    # Wait for left/right mouse click
    while True:
        if mouse.getPressed()[0]:
            return "continue"
        elif mouse.getPressed()[2]:
            return "repeat"
        elif "escape" in event.getKeys():
            save_data_and_exit()
        core.wait(0.1)

# ==================== Task preparation stage ====================
def VWM_stimulation():
    global circle_radius, color_set, positions
    # Predefine 6 positions on the circle (6 evenly spaced)
    angles_deg = [0, 60, 120, 180, 240, 300]
    positions = [(stimulus_radius * math.cos(math.radians(angle)) * 1.1,
                  stimulus_radius * math.sin(math.radians(angle)) * 1.1) for angle in angles_deg]
    # Define a set of colors
    color_set = ["lightblue", "darkgreen", "red", "grey", "lightgreen", "brown", "pink", "blue", "yellow", "purple"]
    # Define circle size (3.5 cm converted to 3.5° visual angle)
    circle_diameter = 3.5
    circle_radius = circle_diameter / 2

def MIT_stimulation():
    global generate_safe_positions, flash_border, Tracking
# Non-overlapping position generation function (for random initial positions)
    def generate_safe_positions(num_stimuli, image_size=(2.5, 2.5), min_dist=4.5):
        image_width, image_height = image_size
        # Calculate screen boundaries (degrees)
        screen_width_deg = win.size[0] / win.monitor.getSizePix()[0] * win.monitor.getWidth()
        screen_height_deg = screen_width_deg * (win.size[1] / win.size[0])
        edge_margin = 0.2  # edge margin
        x_limit = screen_width_deg / 2 - image_width / 2 - edge_margin
        y_limit = screen_height_deg / 2 - image_height / 2 - edge_margin
        positions_list = []
        for _ in range(num_stimuli):
            while True:
                x = random.uniform(-x_limit, x_limit)
                y = random.uniform(-y_limit, y_limit)
                pos = np.array([x, y])
                if all(np.linalg.norm(pos - np.array(p)) > min_dist for p in positions_list):
                    positions_list.append((x, y))
                    break
        return positions_list

    # Border flashing phase (highlight target)
    def flash_border(possible_n=[2, 3, 4, 5], duration=3.0):
        for img in images:
            img.draw()
        n = random.choice(possible_n)
        target_indices = random.sample(range(len(images)), n)
        for idx in target_indices:
            border = visual.Rect(
                win,
                width=images[idx].size[0] + padding,
                height=images[idx].size[1] + padding,
                pos=images[idx].pos,
                lineColor='black',
                lineWidth=4
            )
            border.draw()
    
        # ✅ Start recording
        tracker.start_recording()
        tracker.start_recording()
        win.flip()
        core.wait(duration)
        tracker.stop_recording()
    
        # ✅ Get gaze data
        gaze_events = tracker.get_events().copy()
        fields = [
            "CNT", "TIME",
            "FPOGX", "FPOGY", "FPOGS", "FPOGD", "FPOGID", "FPOGV",
            "LEYEX", "LEYEY", "LEYEZ",
            "LPUPILD", "LPUPILV",
            "REYEX", "REYEY", "REYEZ",
            "RPUPILD", "RPUPILV",
            "BKID", "BKDUR", "BKPMIN",
            "LPMM", "LPMMV", "RPMM", "RPMMV"
        ]
    
        # ✅ Add prefix and encapsulate
        MIT_stim_events_named = [
            {f"MIT_stim_{k}": e.get(k, None) for k in fields}
            for e in gaze_events
        ]
        eye_tool.event_log.append(MIT_stim_events_named)
    
        return target_indices, MIT_stim_events_named

# Target tracking phase (movement)
    def Tracking(duration=7, MIT_AOIs_track=None):
        speed = 6.3  # movement speed (deg/sec)
        image_width, image_height = images[0].size
    
        # Initialize angles and velocity vectors
        angles = [random.uniform(0, 360) for _ in images]
        velocities = [[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in angles]
    
        # Calculate screen boundaries (in deg)
        screen_width_deg = win.size[0] / win.monitor.getSizePix()[0] * win.monitor.getWidth()
        screen_height_deg = screen_width_deg * (win.size[1] / win.size[0])
        x_limit = screen_width_deg / 2 - image_width / 2
        y_limit = screen_height_deg / 2 - image_height / 2
    
        # Initialize positions to current image starting positions (ensure setPos is called externally before calling this function)
        for img in images:
            img.pos = list(img.pos)
    
        # Start gaze recording
        tracker.start_recording()
        timer = core.Clock()
        frame_clock = core.Clock()
    
        while timer.getTime() < duration:
            dt = frame_clock.getTime()
            frame_clock.reset()
    
            # Update each image position
            for i, img in enumerate(images):
                dx = velocities[i][0] * speed * dt
                dy = velocities[i][1] * speed * dt
                new_x = img.pos[0] + dx
                new_y = img.pos[1] + dy
    
                bounced = False
                if abs(new_x) > x_limit:
                    velocities[i][0] *= -1
                    bounced = True
                if abs(new_y) > y_limit:
                    velocities[i][1] *= -1
                    bounced = True
    
                if bounced:
                    angle = np.degrees(np.arctan2(velocities[i][1], velocities[i][0]))
                    angle += random.uniform(-30, 30)
                    rad = np.radians(angle)
                    velocities[i][0] = np.cos(rad)
                    velocities[i][1] = np.sin(rad)
                    new_x = img.pos[0] + velocities[i][0] * speed * dt
                    new_y = img.pos[1] + velocities[i][1] * speed * dt
    
                img.pos = (new_x, new_y)
    
            # Overlap detection and avoidance
            min_dist = image_width
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    pos_i = np.array(images[i].pos)
                    pos_j = np.array(images[j].pos)
                    dist = np.linalg.norm(pos_i - pos_j)
                    if dist < min_dist:
                        for idx in [i, j]:
                            angle = np.degrees(np.arctan2(velocities[idx][1], velocities[idx][0]))
                            angle += random.uniform(150, 210)
                            rad = np.radians(angle)
                            velocities[idx][0] = np.cos(rad)
                            velocities[idx][1] = np.sin(rad)
                        # slightly separate
                        images[i].pos = tuple(pos_i + np.array(velocities[i]) * speed * dt)
                        images[j].pos = tuple(pos_j + np.array(velocities[j]) * speed * dt)
    
            # Draw and refresh
            for img in images:
                img.draw()
            win.flip()
    
        # Stop gaze recording
        tracker.stop_recording()
        gaze_events = tracker.get_events().copy()
        fields = [
            "CNT", "TIME",
            "FPOGX", "FPOGY", "FPOGS", "FPOGD", "FPOGID", "FPOGV",
            "LEYEX", "LEYEY", "LEYEZ",
            "LPUPILD", "LPUPILV",
            "REYEX", "REYEY", "REYEZ",
            "RPUPILD", "RPUPILV",
            "BKID", "BKDUR", "BKPMIN",
            "LPMM", "LPMMV", "RPMM", "RPMMV"
        ]
    
        # Name fields
        MIT_track_events_named = [{f"MIT_track_{k}": e.get(k, None) for k in fields} for e in gaze_events]
        eye_tool.event_log.append(MIT_track_events_named)
        return MIT_track_events_named
    
    # Register nested functions for global use
    globals()['generate_safe_positions'] = generate_safe_positions
    globals()['flash_border'] = flash_border
    globals()['Tracking'] = Tracking

# MIT masking phase function (used in MIT_trial)
def masking(target_MIT, target_index, duration=1.5):
    test_index = target_index if random.random() < 0.5 else random.choice(
        [i for i in range(6) if i != target_index]
    )
    for img in images:
        blocker = visual.Rect(
            win,
            width=img.size[0],
            height=img.size[1],
            pos=img.pos,
            lineColor='black',
            fillColor='black',
            lineWidth=2
        )
        blocker.draw()
    # Draw black border for test image
    mask = visual.Rect(
        win,
        width=images[test_index].size[0] + 0.3,
        height=images[test_index].size[1] + 0.3,
        pos=images[test_index].pos,
        fillColor=None,
        lineColor='black',
        lineWidth=4
    )
    mask.draw()
    win.flip()
    core.wait(duration)
    return test_index

# Helper function: draw masking (for combined task, drawn in two steps)
def masking_only_draw_blockers(win, images, target_index, duration=0.5):
    test_index = target_index if random.random() < 0.5 else random.choice([i for i in range(len(images)) if i != target_index])
    # Draw black squares to cover all images
    for img in images:
        blocker = visual.Rect(win, width=img.size[0], height=img.size[1],
                               pos=img.pos, lineColor='black', fillColor='black', lineWidth=2)
        blocker.draw()
    win.flip()
    core.wait(duration)
    return test_index

def masking_only_draw_mask_border(win, images, test_image):
    # Step 1: redraw all blockers
    for img in images:
        blocker = visual.Rect(
            win,
            width=img.size[0],
            height=img.size[1],
            pos=img.pos,
            lineColor='black',
            fillColor='black',
            lineWidth=2
        )
        blocker.draw()

    # Step 2: highlight the test image border
    highlight = visual.Rect(
        win,
        width=test_image.size[0] + 0.3,
        height=test_image.size[1] + 0.3,
        pos=test_image.pos,
        fillColor=None,
        lineColor='black',
        lineWidth=4
    )
    highlight.draw()

    # Step 3: flip to display
    win.flip()
    core.wait(1.5)

# ==================== VWM Trail ====================
def VWM_sample_display():
    Fixation()

    # ✅ Generate stimulus colors and positions
    sample_colors = random.sample(color_set, 3)
    sample_positions = random.sample(positions, 3)

    # ✅ Begin drawing
    for i in range(3):
        visual.Circle(
            win,
            radius=circle_radius,
            fillColor=sample_colors[i],
            lineColor=sample_colors[i],
            pos=sample_positions[i]
        ).draw()

    # ✅ Record gaze
    tracker.start_recording()
    win.flip()
    core.wait(0.4)
    tracker.stop_recording()

    # ✅ Get gaze data
    gaze_events = tracker.get_events()

    # ✅ Filter fields + add prefix
    fields = [
        "CNT", "TIME",
        "FPOGX", "FPOGY", "FPOGS", "FPOGD", "FPOGID", "FPOGV",
        "LEYEX", "LEYEY", "LEYEZ",
        "LPUPILD", "LPUPILV",
        "REYEX", "REYEY", "REYEZ",
        "RPUPILD", "RPUPILV",
        "BKID", "BKDUR", "BKPMIN",
        "LPMM", "LPMMV", "RPMM", "RPMMV"
    ]
    VWM_stim_events_named = [
        {f"VWM_stim_{k}": e.get(k, None) for k in fields}
        for e in gaze_events
    ]

    return sample_colors, sample_positions, VWM_stim_events_named

def VWM_test_phase_practice(sample_colors, sample_positions):
    #  VWM practice test phase (with instructions)

    # ✅ Add a background to clear the '+' from the Retention interval
    bg = visual.Rect(win, width=999, height=999, fillColor='white', lineColor='white')
    bg.draw()

    test_color = random.choice(sample_colors)
    if random.random() > 0.5:
        test_position = sample_positions[sample_colors.index(test_color)]
        condition = "Match"
    else:
        test_position = random.choice([p for p in sample_positions if p != sample_positions[sample_colors.index(test_color)]])
        condition = "Mismatch"

    test_circle = visual.Circle(win, radius=circle_radius, fillColor=test_color,
                                lineColor=test_color, pos=test_position)
    question = visual.TextStim(win,
        text="Does the circle have the same color and position as before?\nLeft click = Yes, Right click = No",
        color="black", height=0.7, wrapWidth=25, pos=(0, 0)
    )
    test_circle.draw()
    question.draw()
    win.flip()

    rt_clock = core.Clock()
    mouse = event.Mouse()
    while True:
        buttons = mouse.getPressed()
        if buttons[0]:
            rt = rt_clock.getTime(); key = "mouse_left"; break
        elif buttons[2]:
            rt = rt_clock.getTime(); key = "mouse_right"; break
        elif "escape" in event.getKeys():
            print("🚪 Exiting early, saving data collected so far")
            save_data_and_exit()
    return test_color, test_position, condition, key, rt

def VWM_test_phase(sample_colors, sample_positions, VWM_AOIs):
    # Clear remnants from Retention stage
    bg = visual.Rect(win, width=999, height=999, fillColor='white', lineColor='white')
    bg.draw()

    # ✅ Generate test item
    test_color = random.choice(sample_colors)
    if random.random() > 0.5:
        test_position = sample_positions[sample_colors.index(test_color)]
        VWM_condition = "Match"
    else:
        test_position = random.choice([
            p for p in sample_positions
            if p != sample_positions[sample_colors.index(test_color)]
        ])
        VWM_condition = "Mismatch"

    # ✅ Prepare stimulus
    test_circle = visual.Circle(win, radius=circle_radius, fillColor=test_color,
                                lineColor=test_color, pos=test_position)
    question2 = visual.TextStim(win, text="Left = Yes, Right = No",
                                color="black", height=0.7, wrapWidth=25, pos=(0, 0))

    # ✅ Start recording
    tracker.start_recording()
    rt_clock = core.Clock()
    mouse = event.Mouse(win=win)
    mouse.setPos((0, 0))
    mouse.clickReset()

    key = None
    rt = None
    while True:
        test_circle.draw()
        question2.draw()
        win.flip()

        buttons = mouse.getPressed()
        if buttons[0]:
            rt = rt_clock.getTime()
            key = "mouse_left"
            break
        elif buttons[2]:
            rt = rt_clock.getTime()
            key = "mouse_right"
            break
        elif "escape" in event.getKeys():
            print("Exiting early, saving data collected so far")
            save_data_and_exit()
        core.wait(0.001)

    # ✅ Stop recording and extract gaze data
    tracker.stop_recording()
    gaze_events = tracker.get_events().copy()
    print("✅ VWM_test gaze 数:", len(gaze_events))

    # ✅ Add prefix and keep fields
    fields = [
        "CNT", "TIME",
        "FPOGX", "FPOGY", "FPOGS", "FPOGD", "FPOGID", "FPOGV",
        "LEYEX", "LEYEY", "LEYEZ",
        "LPUPILD", "LPUPILV",
        "REYEX", "REYEY", "REYEZ",
        "RPUPILD", "RPUPILV",
        "BKID", "BKDUR", "BKPMIN",
        "LPMM", "LPMMV", "RPMM", "RPMMV"
    ]
    VWM_test_events_named = [
        {f"VWM_test_{k}": e.get(k, None) for k in fields}
        for e in gaze_events
    ]
    eye_tool.event_log.append(VWM_test_events_named)

    return test_color, test_position, VWM_condition, key, rt, {}, VWM_test_events_named

# ==================== MIT trials ====================
def MIT_test_practice(images, target_MIT, test_index, mode="Practice_MIT"):
    # MIT practice test phase (with instructions)
    is_target_masked = (test_index in target_MIT)
    condition = "Match" if is_target_masked else "Mismatch"
    # Set 'NO' button position (using position 0 from circle positions list)
    radius = 5
    angles = np.linspace(90, 90 + 360, 7, endpoint=False)
    positions_list = [(radius * np.cos(np.radians(a)), radius * np.sin(np.radians(a))) for a in angles]
    no_pos = positions_list[0]
    # Randomly assign images to other circle positions
    random.shuffle(positions_list[1:])
    for img, pos in zip(images, positions_list[1:]):
        img.pos = pos

    # Instruction text (keep original prompt unchanged)
    MIT_question_stim = visual.TextStim(
        win,
        text="If the target is in the border,\nclick the picture in the border.\nClick NO if the target is not in the border.",
        pos=(0, 0), height=0.4, color="black", wrapWidth=10
    )
    rt_timer = core.Clock()
    mouse = event.Mouse(win=win)
    mouse.setPos((0, 0))
    clicked = False
    click_pos = None

    # Define function to check if a click is within an image's area (based on image center and size)
    def is_click_inside_image(click_pos, img):
        left = img.pos[0] - (img.size[0] / 2)
        right = img.pos[0] + (img.size[0] / 2)
        bottom = img.pos[1] - (img.size[1] / 2)
        top = img.pos[1] + (img.size[1] / 2)
        return left <= click_pos[0] <= right and bottom <= click_pos[1] <= top

    # Loop until a valid click: only a click on the NO button or any image area is considered valid
    while not clicked:
        for img in images:
            img.draw()
        visual.TextStim(win, text="NO", pos=no_pos, height=1.5, color='black', bold=True).draw()
        MIT_question_stim.draw()
        win.flip()
        # Allow ESC to exit
        keys = event.getKeys()
        if 'escape' in keys:
            print("Experiment terminated by user pressing ESC")
            save_data_and_exit()
        if mouse.getPressed()[0]:
            temp_click_pos = mouse.getPos()
            # If click is on NO button (using distance; still using a simple circular distance check)
            if np.linalg.norm(np.array(no_pos) - np.array(temp_click_pos)) < 1.25:
                click_pos = temp_click_pos
                clicked = True
            else:
                # Check if click is on any image area (use is_click_inside_image)
                for img in images:
                    if is_click_inside_image(temp_click_pos, img):
                        click_pos = temp_click_pos
                        clicked = True
                        break
    rt = rt_timer.getTime()
    # Determine correctness based on condition
    if condition == "Match":
        # If the prompt image is the target, check if the click is within that image's area
        response_correct = is_click_inside_image(click_pos, images[test_index])
    else:
        # If the prompt image is not the target, only clicking NO is correct
        dx = no_pos[0] - click_pos[0]
        dy = no_pos[1] - click_pos[1]
        response_correct = (math.sqrt(dx**2 + dy**2) < 1.25)
    return {
        "reaction_time": rt,
        "is_target_masked": is_target_masked,
        "mouse_click": click_pos,
        "condition": condition,
        "response": response_correct
    }

def MIT_test_phase(images, target_MIT, test_index, mode="MIT"):
    print("✅ Entering MIT_test_phase()")
    # MIT formal test phase
    is_target_masked = (test_index in target_MIT)
    condition = "Match" if is_target_masked else "Mismatch"

    # Place images and 'NO' button
    radius = 5
    angles = np.linspace(90, 450, 7, endpoint=False)
    positions_list = [(radius * np.cos(np.radians(a)), radius * np.sin(np.radians(a))) for a in angles]
    no_pos = positions_list[0]
    random.shuffle(positions_list[1:])
    for img, pos in zip(images, positions_list[1:]):
        img.pos = pos

    # Start recording gaze & response
    tracker.start_recording()
    rt_timer = core.Clock()
    mouse = event.Mouse(win=win)
    mouse.setPos((0, 0))
    mouse.clickReset() 
    clicked = False
    click_pos = None

    def is_click_inside_image(p, img):
        left = img.pos[0] - img.size[0]/2
        right = img.pos[0] + img.size[0]/2
        bottom = img.pos[1] - img.size[1]/2
        top = img.pos[1] + img.size[1]/2
        return left<=p[0]<=right and bottom<=p[1]<=top

    while not clicked:
        for img in images:
            img.draw()
        visual.TextStim(win, text="NO", pos=no_pos, height=1.5, color='black', bold=True).draw()
        win.flip()

        if "escape" in event.getKeys():
            save_data_and_exit()

        if mouse.getPressed()[0]:
            click_pos = mouse.getPos()
            # Click NO or image
            if np.linalg.norm(np.array(no_pos)-np.array(click_pos))<1.25 or any(is_click_inside_image(click_pos, img) for img in images):
                clicked = True

    rt = rt_timer.getTime()
    tracker.stop_recording()

    # Determine correctness
    if condition == "Match":
        correct = is_click_inside_image(click_pos, images[test_index])
    else:
        dx, dy = no_pos[0]-click_pos[0], no_pos[1]-click_pos[1]
        correct = (math.hypot(dx, dy) < 1.25)

    # Keep only raw gaze events
    gaze_events = tracker.get_events().copy()
    fields = [
        "CNT", "TIME",
        "FPOGX", "FPOGY", "FPOGS", "FPOGD", "FPOGID", "FPOGV",
        "LEYEX", "LEYEY", "LEYEZ",
        "LPUPILD", "LPUPILV",
        "REYEX", "REYEY", "REYEZ",
        "RPUPILD", "RPUPILV",
        "BKID", "BKDUR", "BKPMIN",
        "LPMM", "LPMMV", "RPMM", "RPMMV"
    ]
    MIT_test_events_named = [
        {f"MIT_test_{k}": e.get(k, None) for k in fields}
        for e in gaze_events
    ]
    eye_tool.event_log.append(MIT_test_events_named)
    
    return {
        "reaction_time": rt,
        "is_target_masked": is_target_masked,
        "mouse_click": click_pos,
        "condition": condition,
        "response": correct,
        "MIT_test_events": MIT_test_events_named
    }

# Define MIT initialization function
def reset_images(images, size=(2.5, 2.5), min_dist=4.5):
    for img in images:
        img.size = size
    positions_gen = generate_safe_positions(len(images), image_size=size, min_dist=min_dist)
    for i, img in enumerate(images):
        img.pos = positions_gen[i]

# Ensure each MIT target count appears an equal number of times
def generate_balanced_n_sequence(possible_n, total_trials):
    assert total_trials % len(possible_n) == 0, "Total number of trials must be a multiple of the number of target types"
    repeat_each = total_trials // len(possible_n)
    sequence = possible_n * repeat_each
    random.shuffle(sequence)
    return sequence

# Load MIT image stimuli
image_files = [
    r"stimuli/1.jpg", r"stimuli/2.jpg", r"stimuli/3.jpg",
    r"stimuli/4.jpg", r"stimuli/5.jpg", r"stimuli/6.jpg"
]
images = [visual.ImageStim(win, image=img, units='deg') for img in image_files]
padding = 0.01  # border padding

# Initialize data list
data_list = []

# ==================== Trial function definitions ====================
def VWM_trial(is_practice, use_practice_phase=True, phase=""): 
    Trial_begin("1")

    # ✅ Present sample and record gaze
    sample_colors, sample_positions, VWM_stim_events_named = VWM_sample_display()
    Retention_Interval()

    # ✅ Test phase (practice vs formal)
    if use_practice_phase:
        test_color, test_position, VWM_condition, key, rt = VWM_test_phase_practice(sample_colors, sample_positions)
        VWM_test_events_named = []
    else:
        test_color, test_position, VWM_condition, key, rt, _, VWM_test_events_named = VWM_test_phase(sample_colors, sample_positions, [])
    
    # ✅ Determine correctness
    vwm_correct = ((VWM_condition == "Match" and key == "mouse_left") or
                   (VWM_condition == "Mismatch" and key == "mouse_right"))

    # ✅ Construct and save trial data
    data_list.append({
        "phase": phase,
        "task": "VWM",
        "with_instruction": is_practice,
        "VWM_sample_colors": sample_colors,
        "VWM_sample_positions": sample_positions,
        "VWM_test_color": test_color,
        "VWM_test_position": test_position,
        "VWM_condition": VWM_condition,
        "VWM_key": key,
        "VWM_rt": rt,
        "VWM_correct": vwm_correct,
        "VWM_stim_events": VWM_stim_events_named,
        "VWM_test_events": VWM_test_events_named
    })

    core.wait(1.0)
    return vwm_correct

def MIT_trial(n, is_practice, phase=""):
    Trial_begin("2")
    reset_images(images)  # initialize image positions

    # Border cue phase (MIT_stim) and record gaze
    target_indices, MIT_stim_events_named = flash_border(possible_n=[n], duration=3.0)
    
    # Tracking phase (MIT_track) and record gaze
    MIT_track_events_named = Tracking(duration=7) # tracking time

    # Randomly pick one of the flashing targets for testing
    target_index = random.choice(target_indices)

    # Enter masking test phase (MIT_test): draw black blocks + highlight test target
    test_index = masking(target_indices, target_index)

    # Test phase: no distinction between practice and formal
    mit_result = MIT_test_phase(images, target_indices, test_index, mode="MIT")
    test_events_named = mit_result.get("MIT_test_events", [])
    mit_correct = mit_result.get("response", False)

    # Organize behavioral data
    target_filenames = [os.path.basename(images[idx].image) for idx in target_indices]
    clicked_filename = os.path.basename(images[test_index].image) if mit_correct else "NO"
    question_filename = os.path.basename(images[test_index].image)

    # Save this trial data
    data_list.append({
        "phase": phase,
        "task": "MIT",
        "with_instruction": is_practice,
        "MIT_size": n,
        "MIT_target": target_filenames,
        "MIT_question": question_filename,
        "MIT_condition": mit_result.get("condition", ""),
        "MIT_answer": clicked_filename,
        "MIT_correct": mit_correct,
        "MIT_rt": mit_result.get("reaction_time", None),
        "MIT_target_indices": target_indices,
        "MIT_stim_events": MIT_stim_events_named,
        "MIT_track_events": MIT_track_events_named,
        "MIT_test_events": test_events_named  # ✅ use a unified return variable name
    })
    core.wait(1.0)
    return mit_correct

#VWM+MIT trail
def VWM_MIT_trial(n, is_practice, phase=""):
    Trial_begin("1+2")

    # ✅ VWM sample phase: record gaze events (no metrics)
    sample_colors, sample_positions, VWM_stim_events_named = VWM_sample_display()
    reset_images(images)
    Fixation()

    # ✅ MIT stimulus + gaze
    target_indices, MIT_stim_events_named = flash_border(possible_n=[n], duration=3.0)

    # ✅ MIT tracking + gaze
    MIT_track_events_named = Tracking(duration=7)

    target_index = random.choice(target_indices)
    test_index = masking_only_draw_blockers(win, images, target_index)
    test_image = images[test_index]

    # ✅ VWM test phase (keep only gaze events)
    VWM_test_events_named = []
    if is_practice:
        test_color, test_position, VWM_condition, key, rt = VWM_test_phase_practice(sample_colors, sample_positions)
        VWM_test_events_named = []
    else:
        test_color, test_position, VWM_condition, key, rt, _, VWM_test_events_named = VWM_test_phase(sample_colors, sample_positions, [])

    vwm_correct = ((VWM_condition == "Match" and key == "mouse_left") or
                   (VWM_condition == "Mismatch" and key == "mouse_right"))

    # ✅ MIT test phase
    masking_only_draw_mask_border(win, images, test_image)
    if is_practice:
        mit_result = MIT_test_practice(images, target_indices, test_index, mode="Practice_Combo")
        MIT_test_events_named = []
    else:
        mit_result = MIT_test_phase(images, target_indices, test_index, mode="Combo")
        MIT_test_events_named = mit_result.get("MIT_test_events", []) 

    mit_condition = mit_result.get("condition", "")
    mit_key = mit_result.get("mouse_click", None)
    mit_correct = mit_result.get("response", False)
    clicked_filename = os.path.basename(images[test_index].image) if mit_correct else "NO"
    question_filename = os.path.basename(images[test_index].image)
    target_filenames = [os.path.basename(images[idx].image) for idx in target_indices]

    # ✅ Save data (no VWM metrics)
    data_list.append({
        "phase": phase,
        "task": "VWM+MIT",
        "with_instruction": is_practice,
        # VWM behavioral data
        "VWM_sample_colors": sample_colors,
        "VWM_sample_positions": sample_positions,
        "VWM_test_color": test_color,
        "VWM_test_position": test_position,
        "VWM_condition": VWM_condition,
        "VWM_key": key,
        "VWM_rt": rt,
        "VWM_correct": vwm_correct,
        # MIT behavioral data
        "MIT_size": n,
        "MIT_target": target_filenames,
        "MIT_question": question_filename,
        "MIT_condition": mit_condition,
        "MIT_answer": clicked_filename,
        "MIT_correct": mit_correct,
        "MIT_rt": mit_result.get("reaction_time", None),
        "MIT_target_indices": target_indices,
        # gaze event data
        "VWM_stim_events": VWM_stim_events_named,
        "VWM_test_events": VWM_test_events_named,
        "MIT_stim_events": MIT_stim_events_named,
        "MIT_track_events": MIT_track_events_named,
        "MIT_test_events": MIT_test_events_named
    })

    core.wait(1.0)
    return vwm_correct, mit_correct

# ==================== Practice and formal experiment flow ====================
def practice_block(phase='practice'):
    instructions = Instructions(win)
    # Practice Task1 (VWM)
    instructions.VWM_show_instructions()
    while True:
        for _ in range(2):
            VWM_trial(True, use_practice_phase=True, phase=phase)
            last = data_list[-1]
            show_feedback(vwm_correct=last["VWM_correct"])
        decision = confirm_understanding()
        if decision == "continue":
            break
        elif decision == "repeat":
            continue
    for _ in range(2):
        VWM_trial(True, use_practice_phase=False, phase=phase)
        last = data_list[-1]
        show_feedback(vwm_correct=last["VWM_correct"])
    # Practice Task2 (MIT)
    instructions.MIT_show_instructions()
    while True:
        for n in [2, 3]:
            MIT_trial(n, True, phase=phase)
            last = data_list[-1]
            show_feedback(mit_correct=last["MIT_correct"])
        decision = confirm_understanding()
        if decision == "continue":
            break
        elif decision == "repeat":
            continue
    for n in [2, 3]:
        MIT_trial(n, True, phase=phase)
        last = data_list[-1]
        show_feedback(mit_correct=last["MIT_correct"])
    # Practice Task3 (VWM+MIT)
    instructions.VWM_MIT_show_instructions()
    while True:
        for n in [2, 3]:
            VWM_MIT_trial(n, True, phase=phase)
            last = data_list[-1]
            show_feedback(vwm_correct=last["VWM_correct"], mit_correct=last["MIT_correct"])
        decision = confirm_understanding()
        if decision == "continue":
            break
        elif decision == "repeat":
            continue
    for n in [2, 3]:
        VWM_MIT_trial(n, True, phase=phase)
        last = data_list[-1]
        show_feedback(vwm_correct=last["VWM_correct"], mit_correct=last["MIT_correct"])
    core.wait(1.0)

def experiment_block(phase='formal'):
    trial_units = []
    # ✅ VWM: fixed 12 trials (n not needed)
    for _ in range(12):
        trial_units.append(("VWM", None))
    # ✅ MIT: fixed 16 trials, n=2-5 randomly balanced
    mit_ns = [2, 3, 4, 5] * 4  # 4 × 4 = 16
    for n in mit_ns:
        trial_units.append(("MIT", n))
    # ✅ VWM+MIT: 5 trials for each n
    vwm_mit_ns = [2]*5 + [3]*5 + [4]*5 + [5]*5  # total 20 trials
    for n in vwm_mit_ns:
        trial_units.append(("VWM_MIT", n))
    # ✅ Shuffle order
    random.shuffle(trial_units)
    # ✅ Run sequentially
    for trial_type, n in trial_units:
        if trial_type == "VWM":
            VWM_trial(is_practice=False, use_practice_phase=False, phase=phase)
        elif trial_type == "MIT":
            MIT_trial(n=n, is_practice=False, phase=phase)
        elif trial_type == "VWM_MIT":
            VWM_MIT_trial(n=n, is_practice=False, phase=phase)
        core.wait(0.1)

# Main function execution
def main():
    # Check Gazepoint connection
    try:
        with socket.create_connection(("10.56.209.106", 4242), timeout=1.0):
            print("✅ Gazepoint Control online, port open")
    except Exception:
        print("❌ Gazepoint Control not detected, please start Gazepoint software first!")
        core.quit()

    global data_list, eye_tool, tracker
    eye_tool = EyeTracking()
    tracker = eye_tool.tracker
    
    VWM_stimulation()
    MIT_stimulation()

    instructions = Instructions(win)
    #instructions.begin_instructions()

    #practice_block(phase="practice")
    instructions.formal_experiment_instructions()
    experiment_block(phase="formal")
    instructions.end_instructions()
    
    # Save all data
    print("✅ All trials completed")
    os.makedirs("data", exist_ok=True)
    participant = exp_info['Participant']
    base_filename = f"data/{participant}"

    df = pd.DataFrame(data_list)
    df.to_excel(base_filename + ".xlsx", index=False)
    df.to_csv(base_filename + ".csv", index=False)

    # Export gaze event data
    gaze_csv = eye_tool.export_gaze_events_to_csv(base_filename + "_gaze_events.csv")
    gaze_df = pd.read_csv(gaze_csv)
    gaze_df.to_excel(base_filename + "_gaze_events.xlsx", index=False)

    win.close()
    core.quit()

# Run main program
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        print(f"💥 Program crashed：{e}")
        save_data_and_exit()
        raise
