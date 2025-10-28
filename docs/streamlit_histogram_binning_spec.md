# Streamlit Dashboard: Histogram Binning Selector

## Feature Specification

**Location**: Hypothesis Testing / Age Distribution page in Streamlit dashboard
**Component**: Interactive radio button or selectbox for binning strategy
**Purpose**: Educational tool showing how visualization choices affect data interpretation

---

## Three Viewing Modes

### Mode A: Fixed Binning (Standard Practice)
**Selector Label**: "Fixed Bins (Standard)"

**Implementation**:
```python
bins = 30  # Fixed for all diseases
```

**User-Facing Explanation**:
> **What it shows:** Standard statistical practice using the same number of bins (30) for all distributions, regardless of sample size.
>
> **When to use:** When you want direct visual comparison across diseases with consistent granularity. This is the most common approach in scientific publications.
>
> **Reveals:** Overall shape patterns and relative peak positions. Good for comparing diseases with similar sample sizes.
>
> **Limitation:** May create noisy, "choppy" histograms for rare diseases (e.g., Hernia with 227 samples = ~7 per bin). May miss fine details in common diseases with large samples.
>
> **Best for:** General audience, standardized reporting, comparing similar-sized groups.

**Technical Note** (collapsible):
- Used in 95% of published research
- Assumes equal "resolution" appropriate for all datasets
- Simple to explain and reproduce

---

### Mode B: Adaptive Binning (Sample-Size Aware) ⭐ DEFAULT
**Selector Label**: "Adaptive Bins (Recommended)"

**Implementation**:
```python
# Sturges' rule: optimal bins = ⌈log₂(n) + 1⌉
n_with = len(ages_with_disease)
bins = int(np.ceil(np.log2(n_with) + 1))
bins = max(10, min(bins, 30))  # Clamp to reasonable range
```

**User-Facing Explanation**:
> **What it shows:** Intelligently adjusts bin count based on sample size using Sturges' rule (1926). Fewer bins for small samples (reduces noise), more bins for large samples (shows detail).
>
> **When to use:** When comparing diseases with vastly different prevalence (e.g., Hernia: 227 vs. Infiltration: 19,894). This is our default recommendation.
>
> **Reveals:** True underlying age distribution without statistical noise masking the signal. Particularly important for rare diseases where we need to see patterns clearly.
>
> **Example:**
> - Hernia (n=227) → 9 bins (smoother, clearer pattern)
> - Infiltration (n=19,894) → 15 bins (more detail)
>
> **Best for:** Medical researchers, data scientists, comparing diseases with different prevalence rates.

**Technical Note** (collapsible):
- **Sturges' Rule** (1926): `k = ⌈log₂(n) + 1⌉`
- Derived from binomial distribution assumptions
- Optimal for normal distributions
- Widely used in R's `hist()` default
- We clamp to 10-30 range to prevent extremes

**Mathematical Justification**:
```
Sample size (n) → Optimal bins (k)
-----------------------------------------
100             → 8 bins
227 (Hernia)    → 9 bins
1,000           → 11 bins
10,000          → 15 bins
19,894 (Infilt) → 15 bins
100,000         → 18 bins
```

---

### Mode C: Freedman-Diaconis (Data-Distribution Aware)
**Selector Label**: "Freedman-Diaconis (Advanced)"

**Implementation**:
```python
def calculate_bins_fd(data):
    """Freedman-Diaconis rule for optimal bins"""
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:  # Handle edge case
        return 10
    bin_width = 2 * iqr * len(data)**(-1/3)
    n_bins = int(np.ceil((data.max() - data.min()) / bin_width))
    return max(10, min(n_bins, 50))

bins_with = calculate_bins_fd(ages_with_disease)
bins_without = calculate_bins_fd(ages_without_disease)
bins = min(bins_with, bins_without)  # Use smaller to avoid over-binning
```

**User-Facing Explanation**:
> **What it shows:** Advanced method that adapts to BOTH sample size AND data spread (variability). Uses the Interquartile Range (IQR) to determine optimal bin width.
>
> **When to use:** When you suspect the data distributions have different shapes or spreads (e.g., one disease affects wide age range, another is age-specific).
>
> **Reveals:** Fine-grained patterns in the data distribution. More robust to outliers than Sturges' rule. Can detect subtle bimodality (two peaks) or skewness.
>
> **How it works:**
> 1. Calculates IQR (spread of middle 50% of data)
> 2. Wider spread → wider bins (less granularity needed)
> 3. Narrower spread → narrower bins (more detail available)
> 4. Also accounts for sample size like Sturges
>
> **Example:** If elderly patients with Disease X have very consistent ages (narrow IQR), we use fewer, wider bins. If Disease Y spans all ages (wide IQR), we use more bins.
>
> **Best for:** Advanced users, researchers investigating distribution shape, detecting multimodal patterns.

**Technical Note** (collapsible):
- **Freedman-Diaconis Rule** (1981): `bin_width = 2 × IQR × n^(-1/3)`
- More robust to outliers than Sturges (uses IQR, not range)
- Handles skewed distributions better
- Used in modern R packages (`ggplot2`)
- Theoretically optimal for non-normal distributions

**Mathematical Justification**:
```
Dataset characteristics → Optimal bins
-----------------------------------------
Wide IQR, large n     → Many bins (shows detail)
Narrow IQR, large n   → Moderate bins (avoids noise)
Wide IQR, small n     → Fewer bins (avoids empty bins)
Narrow IQR, small n   → Very few bins (shows pattern)
```

---

## UI/UX Design

### Selector Widget
```python
binning_mode = st.radio(
    "Histogram Binning Strategy",
    options=["Fixed Bins (Standard)",
             "Adaptive Bins (Recommended) ⭐",
             "Freedman-Diaconis (Advanced)"],
    index=1,  # Default to Adaptive
    help="Different binning strategies reveal different aspects of the data"
)
```

### Expandable Info Section
```python
with st.expander("ℹ️ Understanding Binning Strategies", expanded=False):
    st.markdown("""
    ### Why does binning matter?

    Histograms divide continuous data (age) into discrete bins. The number
    of bins dramatically affects what patterns you can see:

    - **Too many bins** (for small samples): Noisy, hard to see pattern
    - **Too few bins** (for large samples): Overly smooth, lose detail
    - **Just right**: Depends on your data and question!

    ### Quick Comparison

    | Strategy | Best For | Limitation |
    |----------|----------|------------|
    | Fixed | Standard comparison | Noisy for rare diseases |
    | Adaptive ⭐ | Varied sample sizes | Assumes normal distribution |
    | Freedman-Diaconis | Non-normal data | More complex to explain |

    **Our recommendation**: Start with Adaptive, then explore others to
    understand how visualization choices affect interpretation.
    """)
```

### Side-by-Side Comparison (Advanced Feature)
```python
if st.checkbox("Show side-by-side comparison of all methods"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Fixed")
        # Plot with fixed bins
    with col2:
        st.subheader("Adaptive")
        # Plot with adaptive bins
    with col3:
        st.subheader("Freedman-Diaconis")
        # Plot with FD bins
```

---

## Educational Value

### Learning Objectives
By interacting with these three modes, users will understand:

1. **Visualization is interpretive**: Different valid choices lead to different insights
2. **Sample size matters**: Small samples need different treatment than large ones
3. **Statistical principles**: Sturges' rule, Freedman-Diaconis, IQR concepts
4. **Critical thinking**: "What am I optimizing for?" (smoothness vs. detail)
5. **Domain knowledge**: Medical interpretation requires understanding both statistics AND clinical context

### Key Takeaway Message (displayed below plots)
```python
st.info("""
💡 **Key Insight**: All three methods are "correct" - they just emphasize different aspects:
- **Fixed**: Standardization and reproducibility
- **Adaptive**: Signal-to-noise ratio optimization
- **Freedman-Diaconis**: Distribution shape awareness

The best choice depends on your question: comparing across diseases (Fixed),
understanding individual patterns (Adaptive), or detecting subtle shapes (FD).
""")
```

---

## Implementation Checklist

### Phase 1: Basic Implementation
- [ ] Add radio selector for binning mode
- [ ] Implement all three binning functions
- [ ] Add mode-specific titles to plots
- [ ] Display bin count on each plot
- [ ] Add basic explanatory text

### Phase 2: Educational Enhancements
- [ ] Add expandable "Understanding Binning" section
- [ ] Include mathematical formulas (LaTeX)
- [ ] Show bin calculation for each disease
- [ ] Add "Why this matters" callouts

### Phase 3: Advanced Features
- [ ] Side-by-side comparison mode
- [ ] Interactive bin count slider (override auto-calculation)
- [ ] Export comparison as PDF report
- [ ] Highlight when methods disagree significantly

### Phase 4: Polish
- [ ] Add tooltips on hover showing exact bin counts
- [ ] Animated transitions between modes
- [ ] "Recommended" badge on Adaptive mode
- [ ] Save user preference in session state

---

## Testing Scenarios

### Test Case 1: Rare Disease (Hernia, n=227)
**Expected behavior**:
- Fixed (30 bins): Choppy, many empty bins
- Adaptive (9 bins): Smooth, clear pattern
- FD (~8-12 bins): Similar to adaptive

**Success criteria**: Adaptive and FD show clearer patterns than Fixed

### Test Case 2: Common Disease (Infiltration, n=19,894)
**Expected behavior**:
- Fixed (30 bins): Good detail
- Adaptive (15 bins): Slightly smoother
- FD (15-25 bins): May show more detail if wide age range

**Success criteria**: All three show similar patterns, FD may reveal subtle features

### Test Case 3: Age-Specific Disease (Hernia, narrow age range)
**Expected behavior**:
- FD should suggest fewer bins due to narrow IQR
- Adaptive and Fixed may over-bin

**Success criteria**: FD correctly identifies narrow age clustering

---

## Code Snippet for Streamlit

```python
def plot_age_distribution_interactive(disease, binning_mode):
    """
    Plot age distributions with selectable binning strategy.

    Args:
        disease: Disease name
        binning_mode: "Fixed", "Adaptive", or "Freedman-Diaconis"
    """
    ages_with = metadata_df[metadata_df[disease] == 1]['Patient Age'].dropna()
    ages_without = metadata_df[metadata_df[disease] == 0]['Patient Age'].dropna()

    # Calculate bins based on mode
    if binning_mode == "Fixed Bins (Standard)":
        bins = 30
        method_text = "Fixed bins (n=30)"
    elif binning_mode == "Adaptive Bins (Recommended) ⭐":
        n = len(ages_with)
        bins = int(np.ceil(np.log2(n) + 1))
        bins = max(10, min(bins, 30))
        method_text = f"Adaptive bins (Sturges: n={bins})"
    else:  # Freedman-Diaconis
        bins = calculate_bins_fd(ages_with)
        method_text = f"Freedman-Diaconis (n={bins})"

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(ages_without, bins=bins, density=True, alpha=0.5,
            label=f'Without {disease}', color='lightblue')
    ax.hist(ages_with, bins=bins, density=True, alpha=0.7,
            label=f'With {disease}', color='salmon')

    ax.set_title(f'{disease} Age Distribution\n{method_text}')
    ax.set_xlabel('Age')
    ax.set_ylabel('Density')
    ax.legend()

    st.pyplot(fig)

    # Show diagnostic info
    with st.expander("📊 Binning Details"):
        st.write(f"**Sample sizes:**")
        st.write(f"- With disease: {len(ages_with):,}")
        st.write(f"- Without disease: {len(ages_without):,}")
        st.write(f"\n**Bins used:** {bins}")
        st.write(f"**Samples per bin (with disease):** ~{len(ages_with)/bins:.1f}")
```

---

## References

1. **Sturges, H. A.** (1926). "The Choice of a Class Interval". *Journal of the American Statistical Association*.
2. **Freedman, D. and Diaconis, P.** (1981). "On the histogram as a density estimator: L2 theory". *Probability Theory and Related Fields*.
3. **Scott, D. W.** (1979). "On optimal and data-based histograms". *Biometrika*.

---

## Notes for Development

- Store this as a reusable component: `components/histogram_binning_selector.py`
- Use consistent color scheme (lightblue/salmon) across dashboard
- Ensure accessibility: color-blind safe palette, text alternatives
- Performance: Cache bin calculations for same dataset
- Mobile responsive: Single column layout on small screens
