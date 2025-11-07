# AI Usage in Project Development

## How I Used AI to Build This Project

I extensively leveraged **Claude Code** (Anthropic's AI coding assistant) throughout the entire development lifecycle of the NIH Chest X-Ray Disease Detection project:

**Data Pipeline Development (Notebooks 01-04)**:
- Used AI to generate data collection scripts using the `kagglehub` API for downloading 112,120 chest X-ray images
- AI helped design custom sklearn-compatible transformers for image preprocessing pipelines
- Generated statistical analysis code for hypothesis testing (chi-square, ANOVA, t-tests) with proper interpretation

**Model Development (Notebooks 05-08)**:
- AI assisted in implementing transfer learning architectures (DenseNet121, ResNet50, EfficientNetB3)
- Generated Grad-CAM visualization code for model interpretability
- Created parameterized notebooks compatible with multiple cloud platforms (Kaggle, Google Colab)

**Testing & DevOps**:
- AI wrote 1,363 lines of test code across 47 test functions for notebook validation
- Generated pytest configurations, CI/CD workflows (GitHub Actions)
- Created MLflow integration code for experiment tracking across platforms

**Documentation**:
- AI helped produce 47 technical documentation guides covering architecture, workflows, and troubleshooting
- Generated comprehensive README sections explaining complex ML concepts for multiple audiences
- Created learning objectives verification mapping all 11 assessment criteria to evidence

**Iterative Refinement Process**:
My workflow involved providing AI with specific requirements, reviewing generated code, testing in the actual environment, then iterating with AI to fix issues. For example, when Kaggle P100 GPUs had session time limits, I worked with AI to migrate the training pipeline to Google Colab Pro+ A100, which required adapting data loading, authentication, and result storage mechanisms.

---

## Challenges Faced When Using AI

**1. Platform-Specific Knowledge Gaps**:
AI initially generated Kaggle-compatible code that failed on Google Colab due to different authentication mechanisms (OAuth vs service accounts) and storage APIs (Kaggle Datasets vs Google Cloud Storage). Required multiple iterations to adapt code for cross-platform compatibility.

**2. Medical Domain Accuracy**:
AI sometimes suggested generic computer vision approaches not suitable for medical imaging. For example, aggressive data augmentation (rotation >15°, flipping vertically) that would violate anatomical constraints. I had to verify suggestions against radiology best practices and domain literature.

**3. Outdated API References**:
AI occasionally referenced deprecated APIs (e.g., old Kaggle API vs modern `kagglehub`, TensorFlow 2.x vs 1.x patterns). Required checking current documentation and testing in actual environment to catch these issues.

**4. Over-Optimization Suggestions**:
AI sometimes suggested overly complex solutions when simpler approaches were more appropriate. For instance, proposing ensemble models before establishing a solid baseline, or suggesting Vision Transformers when transfer learning from DenseNet121 was more practical given computational constraints.

**5. Context Window Limitations**:
For large files (e.g., 2000+ line notebooks), AI couldn't always maintain context across the entire file, requiring me to break problems into smaller, focused chunks and manually integrate solutions.

**6. Statistical Interpretation**:
While AI generated statistically correct code, it occasionally misinterpreted medical significance (e.g., statistically significant age correlation doesn't necessarily mean clinically actionable). Required domain judgment to contextualize findings.

---

## How AI Benefited Me and My Project

**1. Accelerated Development (5-10x faster)**:
Tasks that would have taken days (implementing Grad-CAM, setting up MLflow tracking, writing comprehensive tests) were completed in hours with AI assistance. The project evolved from initial concept to production-ready dashboard in approximately 2 weeks instead of an estimated 2+ months.

**2. Best Practices Implementation**:
AI introduced professional development practices I might not have known: Jupytext for notebook version control, pytest-nbmake for testing notebooks, patient-level data stratification to prevent leakage, and proper class weight calculation for imbalanced datasets.

**3. Comprehensive Documentation**:
AI enabled creation of 47 technical guides covering every aspect of the project. This documentation quality (including detailed API references, troubleshooting guides, and platform-specific workflows) would have been prohibitively time-consuming to write manually.

**4. Error Diagnosis & Debugging**:
When encountering cryptic errors (TensorFlow Metal plugin issues, Kaggle API authentication failures, matplotlib backend problems), AI quickly identified root causes and provided targeted solutions, dramatically reducing debugging time.

**5. Cross-Platform Portability**:
AI helped create reusable code that worked across local MacBook M2 Pro, Kaggle P100 GPUs, and Google Colab A100 GPUs with minimal modifications. This flexibility was crucial when Kaggle session limits forced platform migration.

**6. Code Quality & Testing**:
AI generated well-structured, type-annotated code with proper docstrings. The automated test suite (1,363 lines across 47 functions) provided confidence that notebooks worked as expected and caught regressions early.

**7. Learning Catalyst**:
Working with AI exposed me to new libraries (albumentations, timm, mlflow), design patterns (sklearn-compatible transformers, factory patterns for model loading), and ML engineering practices (experiment tracking, reproducible pipelines) that expanded my technical knowledge.

---

## Ethical Considerations in Project Development

**1. Data Privacy & Patient Confidentiality**:
- Used only the publicly available NIH Chest X-Ray dataset with anonymized patient data (no names, no identifiable information)
- Dataset released under CC0 public domain license, ensuring legal and ethical compliance
- Acknowledged in documentation that dataset was collected with IRB approval from NIH Clinical Center

**2. Bias & Fairness Analysis**:
- Conducted explicit statistical analysis of age and gender distributions across disease classes (Notebook 04)
- Documented potential biases: dataset skews older (mean age 46.9 years), predominantly from single institution (NIH Clinical Center)
- Acknowledged in README and dashboard that model performance may not generalize to populations with different demographic characteristics
- Reported per-disease performance transparently, highlighting poor performance on underrepresented diseases (Hernia: 47 cases, AUC 0.453)

**3. Medical Limitations & Disclaimers**:
- **Dashboard disclaimer**: "This system is for research and educational purposes only. It is NOT a medical device and should NOT be used for clinical diagnosis."
- Clearly stated that model requires clinical validation before any real-world deployment
- Referenced need for FDA/CE approval pathway in Future Enhancements (LO6 compliance)
- Acknowledged in executive summary that goal of >0.8 AUC was not achieved for all diseases

**4. Transparency & Reproducibility**:
- Complete code and methodology publicly available on GitHub
- All random seeds fixed for reproducible results (RANDOM_SEED = 42)
- Model training process fully documented with hyperparameters, data splits, and evaluation metrics
- Honest assessment of model limitations, including Grad-CAM analysis showing spurious correlations (model sometimes focuses on non-lung regions)

**5. Algorithmic Accountability**:
- Implemented Grad-CAM explainability to show which image regions influenced predictions
- Created per-disease ROC curves and confusion matrices for transparency
- Documented model errors and failure modes (pneumonia AUC 0.372, challenges with rare diseases)
- Prioritized interpretability over pure performance (chose Grad-CAM over black-box ensemble approaches)

**6. Responsible AI Development**:
- Acknowledged in documentation that medical AI should augment, not replace, radiologist expertise
- Emphasized need for human-in-the-loop validation
- Discussed in Future Enhancements the importance of external validation on diverse datasets (CheXpert, MIMIC-CXR, PadChest) to assess generalization
- Referenced need for continuous learning as new annotated data becomes available

**7. Dual-Use Considerations**:
- Designed system for beneficial use case (assisting radiologists, improving diagnostic efficiency)
- Acknowledged risk of over-reliance on AI predictions without proper clinical validation
- Documented that false negatives could delay treatment, false positives could cause unnecessary procedures
- Emphasized that system performance varies significantly by disease class (AUC range: 0.372-0.755)

**8. Environmental Impact**:
- Acknowledged computational cost of training (Google Colab A100 GPU for 7+ hours)
- Used transfer learning to reduce training time and energy consumption vs training from scratch
- Deployed lightweight model (37MB DenseNet121) suitable for resource-constrained environments

These ethical considerations align with assessment criteria LO6 (ethical considerations, data privacy, governance) and demonstrate awareness of the responsibilities inherent in developing medical AI systems.

---

## Submission Form Answers

### Question 1: Briefly describe how you have used AI to build your project

I extensively used Claude Code (Anthropic's AI assistant) throughout the entire development lifecycle: AI generated data collection scripts using kagglehub API, designed custom sklearn-compatible transformers, created transfer learning implementations (DenseNet121, ResNet50, EfficientNetB3), and implemented Grad-CAM visualization code. AI wrote 1,363 lines of test code across 47 test functions, generated pytest configurations and CI/CD workflows, and produced 47 technical documentation guides. My workflow involved providing AI with specific requirements, testing generated code in the actual environment (local MacBook, Kaggle P100, Google Colab A100), then iterating to fix issues. For example, when Kaggle session limits became problematic, I worked with AI to migrate the training pipeline to Google Colab Pro+ A100, adapting authentication and storage mechanisms.

### Question 2: Briefly describe the challenges you faced when using AI to build your project

Key challenges included: (1) Platform-specific knowledge gaps - AI generated Kaggle code that failed on Colab due to different authentication mechanisms, requiring multiple iterations for cross-platform compatibility; (2) Medical domain accuracy - AI sometimes suggested generic computer vision approaches unsuitable for medical imaging (e.g., aggressive augmentation violating anatomical constraints); (3) Outdated API references - AI occasionally used deprecated APIs requiring manual verification; (4) Over-optimization - AI suggested complex solutions (Vision Transformers, ensembles) when simpler approaches were more practical; (5) Context window limitations - for large 2000+ line notebooks, AI couldn't maintain full context; (6) Statistical interpretation - while AI generated correct code, it sometimes misinterpreted medical significance, requiring domain judgment to contextualize findings.

### Question 3: Briefly describe how AI benefited you and your project

AI accelerated development 5-10x: tasks taking days (Grad-CAM implementation, MLflow setup, comprehensive testing) completed in hours. The project evolved from concept to production-ready dashboard in ~2 weeks vs estimated 2+ months. AI introduced professional practices I might not have known: Jupytext for notebook version control, patient-level stratification to prevent data leakage, proper class weight calculation. AI enabled 47 technical guides with detailed troubleshooting, dramatically reduced debugging time for cryptic errors (TensorFlow Metal issues, authentication failures), and created reusable code working across MacBook M2/Kaggle P100/Colab A100. The automated test suite (1,363 lines) provided confidence notebooks worked correctly, and working with AI exposed me to new libraries and ML engineering practices that expanded my technical knowledge.

### Question 4: Explain the ethical considerations taken into account when building your project

Data Privacy: Used only publicly available NIH dataset with anonymized patient data under CC0 license, acknowledging IRB approval. Bias Analysis: Conducted explicit statistical analysis of age/gender distributions, documented dataset biases (skews older, single institution), reported poor performance on underrepresented diseases transparently. Medical Limitations: Dashboard includes clear disclaimer "for research/educational purposes only, NOT a medical device," stated need for clinical validation and FDA/CE approval before deployment. Transparency: Complete code on GitHub, fixed random seeds for reproducibility, honest assessment of limitations including Grad-CAM showing spurious correlations. Algorithmic Accountability: Implemented Grad-CAM explainability, per-disease ROC curves, documented failure modes. Responsible Development: Emphasized AI should augment not replace radiologist expertise, need for human-in-the-loop validation, external validation on diverse datasets. Acknowledged risks of false negatives (delayed treatment) and false positives (unnecessary procedures), and computational/environmental costs of GPU training.
