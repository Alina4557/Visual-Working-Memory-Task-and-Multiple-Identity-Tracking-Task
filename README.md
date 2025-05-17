# Visual Working Memory + Multiple Identity Tracking Task (Behavioral Experiment with/without Eye Tracking)

--------------------

## 🧪 Experiment Overview

This experiment investigates how attentional load from Multiple Identity Tracking (MIT) affects Visual Working Memory (VWM).

The VWM task is based on: Fougnie & Marois (2006)  
The MIT task is based on: Oksama & Hyönä (2016)

This script contains three task conditions:  
- VWM-only  
- MIT-only  
- VWM+MIT dual task  

Each participant completes 48 randomized trials:  
- 12 VWM-only trials  
- 16 MIT-only trials  
- 20 dual-task (VWM+MIT) trials  
- MIT task includes four target sizes (2, 3, 4, 5). Each target size is performed 5 times in both MIT-only and VWM+MIT conditions.

--------------------

## 🧰 Script Version

- Behavioral-only script includes both practice and formal trials.  
- Behavioral + Eye-tracking script includes only formal trials (practice not included in main loop).

--------------------

## 🖥️ Software

- PsychoPy (tested with version 2024)  
- Gazepoint Control (tested with version 7.0.0)

--------------------

## 🧾 Data Output

All behavioral responses are recorded with PsychoPy, and gaze events are recorded using Gazepoint GP3 HD and stored in separate files as:  
- `.xlsx` Excel file  
- `.csv` backup file

--------------------

## 📂 File Structure

```
project_folder/
├── stimuli/                                 # Experimental images (MIT task)
├── VWM+MIT (behavioral experiment only).py  # Behavior-only script
├── VWM+MIT (behavioral experiment+ Eyetracking).py  # Behavior + Eye-tracking version
└── README.md                                # This file
```

--------------------

## 👩‍💻 Authors

- Experiment programming and script development: Cuiwei Lu  
- Experimental collaborators: Zhiyi Hu, Yunwei Han

There are no publications and no conflicts of interest.

--------------------

## 📚 Reference

Fougnie, D., & Marois, R. (2006). Distinct capacity limits for attention and working memory: Evidence from attentive tracking and visual working memory paradigms. *Psychological Science, 17*(6), 526–534. https://doi.org/10.1111/j.1467-9280.2006.01739.x  

Oksama, L., & Hyönä, J. (2016). Position tracking and identity tracking are separate systems: Evidence from eye movements. *Cognition, 146*, 393–409. https://doi.org/10.1016/j.cognition.2015.10.016
