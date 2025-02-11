import pandas as pd

# -------------------------------------------------------------------------------------------------
# google blazepose data
# -------------------------------------------------------------------------------------------------

# Function to calculate averages for specific categories
def calculate_category_averages(df, competition, gender=None):
    filtered_df = df[df['competition'] == competition]
    if gender:
        filtered_df = filtered_df[filtered_df['gender'] == gender]
    return filtered_df[['pose_detection_percentage', 'fps']].mean()

blazepose_data = pd.read_pickle('google_blazepose_data.pd')

# Calculate averages for each category
# Indoor Lenzburg - Overall
lenzburg_indoor_avg = calculate_category_averages(blazepose_data, 'lenzburg')

# Female Indoor Lenzburg
female_lenzburg_indoor_avg = calculate_category_averages(blazepose_data, 'lenzburg', 'female')

# Male Indoor Lenzburg
male_lenzburg_indoor_avg = calculate_category_averages(blazepose_data, 'lenzburg', 'male')

# Outdoor Villars - Overall
villars_outdoor_avg = calculate_category_averages(blazepose_data, 'villars')

# Female Outdoor Villars
female_villars_outdoor_avg = calculate_category_averages(blazepose_data, 'villars', 'female')

# Male Outdoor Villars
male_villars_outdoor_avg = calculate_category_averages(blazepose_data, 'villars', 'male')

# Output the results in the specified format
print(
    f"Lenzburg Indoor Overall Average Pose Detection blazepose: {lenzburg_indoor_avg['pose_detection_percentage']:.2f}% | Average FPS: {lenzburg_indoor_avg['fps']:.2f}")
print(
    f"Female Indoor Lenzburg Average Pose Detection blazepose: {female_lenzburg_indoor_avg['pose_detection_percentage']:.2f}% | Average FPS: {female_lenzburg_indoor_avg['fps']:.2f}")
print(
    f"Male Indoor Lenzburg Average Pose Detection blazepose: {male_lenzburg_indoor_avg['pose_detection_percentage']:.2f}% | Average FPS: {male_lenzburg_indoor_avg['fps']:.2f}")

print(
    f"Villars Outdoor Overall Average Pose Detection blazepose: {villars_outdoor_avg['pose_detection_percentage']:.2f}% | Average FPS: {villars_outdoor_avg['fps']:.2f}")
print(
    f"Female Outdoor Villars Average Pose Detection blazepose: {female_villars_outdoor_avg['pose_detection_percentage']:.2f}% | Average FPS: {female_villars_outdoor_avg['fps']:.2f}")
print(
    f"Male Outdoor Villars Average Pose Detection blazepose: {male_villars_outdoor_avg['pose_detection_percentage']:.2f}% | Average FPS: {male_villars_outdoor_avg['fps']:.2f}")

# -------------------------------------------------------------------------------------------------
# RTMO data
# -------------------------------------------------------------------------------------------------

# Function to calculate averages for specific categories
def calculate_category_averages(df, competition, gender=None):
    filtered_df = df[df['competition'] == competition]
    if gender:
        filtered_df = filtered_df[filtered_df['gender'] == gender]
    return filtered_df[['pose_detection_percentage', 'fps']].mean()

rtmo_data = pd.read_pickle('rtmo_data.pd')

# Calculate averages for each competition and category
# Indoor Lenzburg
lenzburg_indoor_avg = calculate_category_averages(rtmo_data, 'lenzburg')
female_lenzburg_indoor_avg = calculate_category_averages(rtmo_data, 'lenzburg', 'female')
male_lenzburg_indoor_avg = calculate_category_averages(rtmo_data, 'lenzburg', 'male')

# Outdoor Villars
villars_outdoor_avg = calculate_category_averages(rtmo_data, 'villars')
female_villars_outdoor_avg = calculate_category_averages(rtmo_data, 'villars', 'female')
male_villars_outdoor_avg = calculate_category_averages(rtmo_data, 'villars', 'male')

# Output the results in the specified format
print(
    f"Lenzburg Indoor Overall Average Pose Detection RTMO: {lenzburg_indoor_avg['pose_detection_percentage']:.2f}% | Average FPS: {lenzburg_indoor_avg['fps']:.2f}")
print(
    f"Female Indoor Lenzburg Average Pose Detection RTMO: {female_lenzburg_indoor_avg['pose_detection_percentage']:.2f}% | Average FPS: {female_lenzburg_indoor_avg['fps']:.2f}")
print(
    f"Male Indoor Lenzburg Average Pose Detection RTMO: {male_lenzburg_indoor_avg['pose_detection_percentage']:.2f}% | Average FPS: {male_lenzburg_indoor_avg['fps']:.2f}")

print(
    f"Villars Outdoor Overall Average Pose Detection RTMO: {villars_outdoor_avg['pose_detection_percentage']:.2f}% | Average FPS: {villars_outdoor_avg['fps']:.2f}")
print(
    f"Female Outdoor Villars Average Pose Detection RTMO: {female_villars_outdoor_avg['pose_detection_percentage']:.2f}% | Average FPS: {female_villars_outdoor_avg['fps']:.2f}")
print(
    f"Male Outdoor Villars Average Pose Detection RTMO: {male_villars_outdoor_avg['pose_detection_percentage']:.2f}% | Average FPS: {male_villars_outdoor_avg['fps']:.2f}")
