import matplotlib.pyplot as plt
import numpy as np

# Your actual project data
sprints = ['Sprint 1 - 100%', 'Sprint 2 - 86%', 'Sprint 3 - 100%', 'Sprint 4 - 73%']
committed = [30, 28, 25, 15] # Estimated points based on your timeline
completed = [30, 24, 25, 11] # Sprint 4 reflects the dropped Cloud Deployment (11/15 days)

x = np.arange(len(sprints))
width = 0.35

# Creating the plot
fig, ax = plt.subplots(figsize=(8, 5))

# Using the exact teal and dark blue colors from your sample
rects1 = ax.bar(x - width/2, committed, width, label='Committed', color='#11999e')
rects2 = ax.bar(x + width/2, completed, width, label='Completed', color='#0b3c5d')

# Formatting to match the sample style perfectly
ax.set_title('Committed vs. Completed', fontsize=24, fontweight='bold', color='#1a1a3a', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(sprints, fontsize=11)
ax.set_ylim(0, 35)

# Removing top and right borders (Spines)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')

# Adding horizontal grid lines
ax.yaxis.grid(True, color='#e0e0e0', linestyle='-', linewidth=1)
ax.set_axisbelow(True) # Puts grid lines behind the bars

# Customizing the legend to sit at the bottom
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=14, prop={'weight':'bold'})

plt.tight_layout()
plt.show()