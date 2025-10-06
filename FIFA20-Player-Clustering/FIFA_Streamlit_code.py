import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Sidebar: Cluster Information
# -------------------------------
st.sidebar.title("📘 Cluster Interpretations")
st.sidebar.markdown("""
### ⚽ **Cluster 1: Attacking & Skill-Oriented Players**
- High in **attacking**, **dribbling**, and **ball control**
- Low in **goalkeeping** attributes  
- Typical Positions: **ST, CF, CAM, LW, RW, LM, RM**
- **Strengths:** Creativity, agility, scoring ability  
- **Weaknesses:** Defensive contribution  

> Represents *offensive and creative players* focused on scoring and playmaking.

---

### 🧤 **Cluster 2: Defensive & Goalkeeping Players**
- High in **goalkeeping** and **defensive** attributes  
- Low in **dribbling** and **attacking** skills  
- Typical Positions: **GK, CDM, CB, LWB, RWB**
- **Strengths:** Reflexes, positioning, handling  
- **Weaknesses:** Offensive ability  

> Represents *defensive and goalkeeping specialists* focused on protecting the goal.
""")

# -------------------------------
# Sidebar: Input Feature Descriptions
# -------------------------------
feature_info = {
    'height_cm': 'Player height in centimeters (140-220)',
    'attacking_heading_accuracy': 'Ability to score with headers (0-100)',
    'skill_dribbling': 'Dribbling skill (0-100)',
    'skill_ball_control': 'Ball control skill (0-100)',
    'movement_balance': 'Player balance while moving (0-100)',
    'goalkeeping_diving': 'Goalkeeper diving ability (0-100)',
    'goalkeeping_handling': 'Goalkeeper handling ability (0-100)',
    'goalkeeping_kicking': 'Goalkeeper kicking power (0-100)',
    'goalkeeping_positioning': 'Goalkeeper positioning (0-100)',
    'goalkeeping_reflexes': 'Goalkeeper reflexes (0-100)',
    'ls': 'Left Striker shooting skill (0-100)',
    'st': 'Striker shooting skill (0-100)',
    'rs': 'Right Striker shooting skill (0-100)',
    'lw': 'Left Wing player skill (0-100)',
    'lf': 'Left Forward player skill (0-100)',
    'cf': 'Center Forward player skill (0-100)',
    'rf': 'Right Forward player skill (0-100)',
    'rw': 'Right Wing player skill (0-100)',
    'lam': 'Left Attacking Midfielder skill (0-100)',
    'cam': 'Central Attacking Midfielder skill (0-100)',
    'ram': 'Right Attacking Midfielder skill (0-100)',
    'lm': 'Left Midfielder skill (0-100)',
    'lcm': 'Left Center Midfielder skill (0-100)',
    'cm': 'Central Midfielder skill (0-100)',
    'rcm': 'Right Center Midfielder skill (0-100)',
    'rm': 'Right Midfielder skill (0-100)',
    'lwb': 'Left Wing Back skill (0-100)',
    'ldm': 'Left Defensive Midfielder skill (0-100)',
    'cdm': 'Central Defensive Midfielder skill (0-100)',
    'rwb': 'Right Wing Back skill (0-100)'
}

st.sidebar.title("📌 Input Features Description")
st.sidebar.write("Enter player attributes based on the descriptions and ranges below:")

for feature, desc in feature_info.items():
    st.sidebar.markdown(f"**{feature}**: {desc}")

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Enter player attributes to predict which cluster (player type) they belong to.")

# -------------------------------
# Load Pretrained Objects
# -------------------------------
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
model = joblib.load("kmeans.pkl")

# -------------------------------
# Streamlit App Title
# -------------------------------
st.title("⚽ FIFA 20 Player Clustering App")
st.write("""
This app predicts **FIFA 20 Player Cluster** based on player attributes.
Enter the player’s skill and goalkeeping attributes below:
""")

# -------------------------------
# Input Features
# -------------------------------
height_cm = st.number_input("Height (cm)", min_value=140, max_value=220, value=175)
attacking_heading_accuracy = st.number_input("Attacking Heading Accuracy", min_value=0, max_value=100, value=50)
skill_dribbling = st.number_input("Skill Dribbling", min_value=0, max_value=100, value=50)
skill_ball_control = st.number_input("Skill Ball Control", min_value=0, max_value=100, value=50)
movement_balance = st.number_input("Movement Balance", min_value=0, max_value=100, value=50)
goalkeeping_diving = st.number_input("Goalkeeping Diving", min_value=0, max_value=100, value=50)
goalkeeping_handling = st.number_input("Goalkeeping Handling", min_value=0, max_value=100, value=50)
goalkeeping_kicking = st.number_input("Goalkeeping Kicking", min_value=0, max_value=100, value=50)
goalkeeping_positioning = st.number_input("Goalkeeping Positioning", min_value=0, max_value=100, value=50)
goalkeeping_reflexes = st.number_input("Goalkeeping Reflexes", min_value=0, max_value=100, value=50)

ls = st.number_input("LS", min_value=0.0, max_value=100.0, value=50.0)
st_ = st.number_input("ST", min_value=0.0, max_value=100.0, value=50.0)
rs = st.number_input("RS", min_value=0.0, max_value=100.0, value=50.0)
lw = st.number_input("LW", min_value=0.0, max_value=100.0, value=50.0)
lf = st.number_input("LF", min_value=0.0, max_value=100.0, value=50.0)
cf = st.number_input("CF", min_value=0.0, max_value=100.0, value=50.0)
rf = st.number_input("RF", min_value=0.0, max_value=100.0, value=50.0)
rw = st.number_input("RW", min_value=0.0, max_value=100.0, value=50.0)
lam = st.number_input("LAM", min_value=0.0, max_value=100.0, value=50.0)
cam = st.number_input("CAM", min_value=0.0, max_value=100.0, value=50.0)
ram = st.number_input("RAM", min_value=0.0, max_value=100.0, value=50.0)
lm = st.number_input("LM", min_value=0.0, max_value=100.0, value=50.0)
lcm = st.number_input("LCM", min_value=0.0, max_value=100.0, value=50.0)
cm = st.number_input("CM", min_value=0.0, max_value=100.0, value=50.0)
rcm = st.number_input("RCM", min_value=0.0, max_value=100.0, value=50.0)
rm = st.number_input("RM", min_value=0.0, max_value=100.0, value=50.0)
lwb = st.number_input("LWB", min_value=0.0, max_value=100.0, value=50.0)
ldm = st.number_input("LDM", min_value=0.0, max_value=100.0, value=50.0)
cdm = st.number_input("CDM", min_value=0.0, max_value=100.0, value=50.0)
rwb = st.number_input("RWB", min_value=0.0, max_value=100.0, value=50.0)

# -------------------------------
# Combine Inputs
# -------------------------------
features = np.array([[height_cm, attacking_heading_accuracy, skill_dribbling, skill_ball_control,
                      movement_balance, goalkeeping_diving, goalkeeping_handling, goalkeeping_kicking,
                      goalkeeping_positioning, goalkeeping_reflexes, ls, st_, rs, lw, lf, cf, rf, rw,
                      lam, cam, ram, lm, lcm, cm, rcm, rm, lwb, ldm, cdm, rwb]])

# -------------------------------
# Cluster Descriptions
# -------------------------------
cluster_description = {
    0: "⚽ Attacking & Skill-Oriented Player: Strong in dribbling, ball control, and attacking; typically forwards or midfielders.",
    1: "🧤 Defensive & Goalkeeping Player: Strong in handling, reflexes, and positioning; typically goalkeepers or defenders."
}

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Cluster"):
    try:
        # Apply preprocessing
        scaled = scaler.transform(features)
        reduced = pca.transform(scaled)

        # Predict
        cluster = model.predict(reduced)[0]

        # Show result
        st.success(f"🏆 This player belongs to **Cluster {cluster}**")
        st.info(cluster_description[cluster])

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
