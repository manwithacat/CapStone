"""
Radiology Guide - Full-page educational resource for chest X-ray interpretation

Separate view from main dashboard with its own sidebar navigation.
"""

import pathlib
from typing import Optional, Dict
import streamlit as st
from src.utils.image_modal import render_clickable_xray


def render_radiology_guide_tab(df=None, disease_colors: Optional[Dict[str, str]] = None):
    """
    Render the full-page Radiology Guide view.

    Args:
        df: Main dataset (optional)
        disease_colors: Color mapping for diseases (optional)
    """

    st.markdown("# 📚 Radiology Guide: Chest X-Ray Interpretation")

    st.info("""
    **Educational Resource for Healthcare Professionals and Students**

    This guide provides foundational knowledge for interpreting chest X-rays and understanding
    the AI-powered disease detection system. It is intended for educational purposes only.
    """)

    # ============================================================================
    # SIDEBAR - Navigation for radiology guide
    # ============================================================================
    
    # Back to dashboard button at top
    if st.sidebar.button("← Back to Dashboard", width='stretch', type="primary"):
        st.session_state.view = 'dashboard'
        st.query_params['view'] = 'dashboard'
        st.rerun()

    st.sidebar.markdown("### Navigation")
    st.sidebar.markdown("""
    - [Introduction](#introduction)
    - [Normal Anatomy](#normal-anatomy)
    - [Pathology Guide](#pathology-guide)
    - [Reading Strategy](#reading-strategy)
    - [AI in Radiology](#ai-in-radiology)
    """)

    # ============================================================================
    # MAIN CONTENT
    # ============================================================================

    # Define figures_dir for use in pathology section
    figures_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "outputs" / "figures"

    # Introduction Section
    st.markdown('<div id="introduction"></div>', unsafe_allow_html=True)
    st.markdown("## 📖 Introduction to Chest X-Rays")
    st.markdown("""
    ### What is a Chest X-Ray?

    A chest X-ray is one of the oldest and most commonly used medical imaging tests. If you've ever had
    a persistent cough, chest pain, or breathing difficulties, chances are your doctor ordered a chest X-ray.
    This simple test uses a small amount of radiation (about the same as you'd naturally absorb during a
    short airplane flight) to create detailed pictures of your heart, lungs, airways, blood vessels, and the
    bones in your chest and spine.

    Think of an X-ray as a shadow photograph. Dense structures like bones block most of the X-rays and appear
    white on the image, while air-filled lungs let X-rays pass through easily and appear dark or black. Your
    heart, blood vessels, and other tissues show up as various shades of gray in between. This creates a detailed
    map that doctors can use to spot problems.

    Chest X-rays are incredibly versatile diagnostic tools. Doctors use them to diagnose lung infections like
    pneumonia, spot lung cancer or tuberculosis, check if your heart is enlarged, look for broken ribs after an
    injury, or monitor chronic lung diseases like COPD or cystic fibrosis. They're also commonly used before
    surgeries to ensure your lungs and heart are healthy enough for the procedure.

    ### Understanding Different X-Ray Views

    When you get a chest X-ray, the technician might ask you to stand or sit in different positions. This isn't
    random - different angles reveal different information, like taking photos of a building from multiple sides
    to see its complete structure.

    The **PA (posteroanterior) view** is the standard chest X-ray you'll usually get. You stand facing a flat
    plate with your chest pressed against it, and the X-ray machine is positioned behind you. The beam passes
    from your back (posterior) to your front (anterior) - hence the name "posteroanterior" or PA. This positioning
    is preferred because it places your heart closer to the detector, which minimizes distortion and gives doctors
    the most accurate picture of your heart size. You'll be asked to take a deep breath and hold it so your lungs
    are fully expanded, making it easier to spot any abnormalities.

    Sometimes, especially if you're too ill to stand or you're in a hospital bed, they'll take an **AP
    (anteroposterior) view** instead. This is essentially the reverse - the X-ray plate goes behind you and the
    machine shoots from the front. While this is more convenient when you can't stand, it does make your heart
    look bigger than it really is because it's farther from the detector. Radiologists know this and make
    allowances when reading AP films, but PA views are generally more accurate.

    A **lateral view** is like taking a picture of your chest from the side. You'll stand with your side against
    the plate, usually with your arms raised. This view is particularly helpful for pinpointing exactly where
    something unusual is located - is that shadow in the front or back of your lung? The lateral view provides
    that depth information that a front-facing X-ray can't show.
    """)

    st.markdown("---")

    # Normal Anatomy Section
    st.markdown('<div id="normal-anatomy"></div>', unsafe_allow_html=True)
    st.markdown("## 🫀 Normal Anatomy")

    st.markdown("""
    Understanding what a normal chest X-ray looks like is the first step in recognizing when something is wrong.
    Let's take a tour through your chest, starting with the most prominent features you'll see on an X-ray.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Your Heart and the Central Chest

        The heart appears as a prominent white (opaque) shape in the center-left of your chest. On a normal PA chest
        X-ray, your heart shouldn't be wider than half the width of your entire chest. Doctors measure this by drawing
        a line across your chest at its widest point and another across your heart at its widest point. If the heart
        line is more than 50% of the chest line, it suggests your heart might be enlarged (cardiomegaly) - a sign that
        it might be working too hard due to conditions like high blood pressure, heart valve problems, or heart failure.

        The area in the middle of your chest, between your lungs, is called the mediastinum (pronounced "mee-dee-ah-STY-num").
        This space houses not just your heart, but also your major blood vessels, trachea (windpipe), esophagus (food pipe),
        and several important lymph nodes. The largest blood vessel in your body, the aorta, arches over your heart like a
        candy cane - you can actually see a bump called the "aortic knob" on the left side of your upper chest on an X-ray.

        Your heart has distinct borders that radiologists look for. On the right side, you can see the superior vena cava
        (the large vein bringing blood from your upper body), your right atrium (the upper right chamber of your heart),
        and where it connects to the inferior vena cava (bringing blood from your lower body). On the left side, that aortic
        knob we mentioned sits at the top, below that is your pulmonary artery (carrying blood to your lungs), and the
        prominent curved left border is your left ventricle - the powerhouse chamber that pumps blood to your entire body.

        ### Your Lungs and Airways

        Your lungs are the large, dark (almost black) areas on either side of your heart. They appear dark because they're
        filled with air, which lets X-rays pass right through. Healthy lungs should look symmetrical - the left and right
        should be roughly mirror images of each other, though the right lung is slightly larger because your heart takes up
        more space on the left side.

        Within these dark lung fields, you'll see branching white lines extending from the center outward, like tree branches.
        These are your blood vessels carrying blood to and from your lungs to pick up oxygen and release carbon dioxide.
        In a healthy lung, these vessels should gradually get smaller and fainter as they extend toward the edges of your lungs.
        If they look unusually prominent or thick, it might indicate increased blood pressure in the lungs or excess fluid.

        Your trachea (windpipe) appears as a dark column running down the center of your neck into your chest. At about the
        level of your collarbone, it splits into two main bronchi - the left and right airways that feed into each lung. This
        split point is called the carina, and radiologists use it as a landmark when reading X-rays.

        Doctors often divide your lungs into three zones for easier description: upper zones (above your second rib), mid zones
        (between your second and fourth ribs), and lower zones (below your fourth rib). This helps them precisely communicate
        where they see something unusual, like saying "there's an opacity in the right upper zone."
        """)

    with col2:
        st.markdown("""
        ### Your Bones and Supporting Structures

        Even though chest X-rays are primarily for looking at your heart and lungs, the bones of your chest provide a
        detailed frame around these organs, and problems with these bones can be important findings.

        You have twelve pairs of ribs that wrap around your chest like protective bands. When radiologists count ribs,
        they count from the back (posterior) because the ribs are more distinct there. The upper ribs (1-3) are thick
        and stout, protecting vital structures like the major blood vessels. The middle ribs (4-9) are the ones most
        likely to break if you fall or have a chest impact. The lower ribs (10-12) are called "floating ribs" because
        they don't connect to your breastbone - they're shorter and more flexible.

        Rib fractures show up as breaks or irregularities in the smooth curves of the ribs. Sometimes you'll see a bump
        along a rib - this is callus formation, which is how your body heals a fracture by laying down new bone. While
        a single rib fracture is painful but not usually dangerous, multiple broken ribs can indicate serious trauma and
        can potentially puncture a lung or injure other organs.

        Your clavicles (collarbones) appear as thin, curved bones running horizontally near the top of your chest. On a
        properly positioned X-ray, they should look symmetrical. These bones are notorious for breaking during falls,
        especially in children and young adults. A broken clavicle usually shows up clearly as a discontinuity in the bone,
        sometimes with the fragments overlapping.

        Your spine runs vertically down the center-back of your chest. On a frontal X-ray, it should appear as a straight
        column. If it's curved or twisted to one side, this could indicate scoliosis. The individual vertebrae should have
        normal height and density. Compression fractures, often seen in people with osteoporosis, cause one or more vertebrae
        to collapse and appear wedge-shaped rather than rectangular.

        Finally, various soft tissues surround these structures. In women, breast tissue appears as soft shadows over the
        lower lung fields - radiologists learn to mentally subtract this normal finding when reading the X-ray. The muscles
        of your chest wall and the skin itself create subtle shadows that experienced radiologists learn to ignore while
        focusing on the important structures underneath.
        """)

    st.markdown("---")

    # Pathology Guide Section
    st.markdown('<div id="pathology-guide"></div>', unsafe_allow_html=True)
    st.markdown("## 🔬 Pathology Guide (14 Conditions)")

    st.markdown("""
    The NIH dataset includes 14 thoracic pathology labels. Understanding these conditions is essential
    for interpreting the AI's predictions and validating clinical findings.
    """)

    render_pathology_section(figures_dir)

    st.markdown("---")

    # Reading Strategy Section
    st.markdown('<div id="reading-strategy"></div>', unsafe_allow_html=True)
    st.markdown("## 📋 How Doctors Read Chest X-Rays")

    st.markdown("""
    Have you ever wondered how radiologists look at chest X-rays without missing important details? They don't just
    glance at the image and hope to spot something unusual. Instead, they use a methodical system that ensures they
    check every structure in the same order, every time. Think of it like a pilot's pre-flight checklist - following
    the same routine means nothing gets overlooked, even on a busy day after reading dozens of X-rays.

    The most popular system is called the ABCDE method. It's an easy-to-remember acronym that guides doctors through
    examining each major system in your chest, from your airways down to your bones and everything in between.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### A - Airway

        First, radiologists check your airway - the path air takes to reach your lungs. They start by looking at your
        trachea (windpipe), that dark column running down the center of your chest. It should run straight down the
        middle, like the center line on a highway. If it's pushed to one side, something might be pushing it - perhaps
        a mass, an enlarged lymph node, or a collapsed lung pulling it over. They also check the carina, the point where
        your trachea splits into the left and right bronchi. This branching point normally looks like an upside-down Y
        at about the level where your fourth rib meets your breastbone.

        ### B - Breathing (Your Lungs)

        Next comes the most important part - your lungs themselves. Radiologists systematically scan each lung field,
        looking for areas that appear too white (suggesting fluid, infection, or solid masses) or too black (suggesting
        too much air or a collapsed lung). They check that both lungs look symmetrical and that the blood vessel patterns
        look normal - gradually branching and fading as they reach the outer edges of your lungs.

        They pay special attention to the hilum (pronounced "HIGH-lum") on each side - this is where the main bronchi and
        major blood vessels enter your lungs. These areas should look similar on both sides. They also examine the pleura,
        the thin membrane lining your lungs, looking for signs of fluid accumulation or thickening. Finally, they check your
        diaphragm - the dome-shaped muscle that sits below your lungs. It should form smooth, sharp angles where it meets
        your rib cage (the costophrenic angles). Blunted angles often indicate fluid collecting in the space around your lungs.

        ### C - Circulation (Heart and Blood Vessels)

        The "circulation" check focuses on your heart and major blood vessels. Radiologists measure your heart size using
        that cardiothoracic ratio we discussed earlier - your heart should be less than half the width of your chest. They
        trace the borders of your heart, making sure they're smooth and sharp. Fuzzy or indistinct borders can indicate
        fluid around the heart or an infection in the lung tissue right next to it.
        """)

    with col2:
        st.markdown("""
        They also look at the aorta (your body's main artery), checking that the aortic knob isn't enlarged or oddly shaped,
        which could suggest an aneurysm - a dangerous weakening and bulging of the artery wall. The pulmonary vessels - the
        arteries and veins in your lungs - should have a characteristic pattern. If the upper lobe vessels look unusually
        prominent compared to the lower lobe vessels, it might indicate that your heart isn't pumping effectively and blood
        is backing up into your lungs.

        ### D - Disability (Bones and Alignment)

        The "disability" check involves examining all the bones visible on the X-ray for fractures, lesions, or
        abnormalities. Radiologists systematically look at each rib, following its curve from back to front, checking for
        breaks or unusual bumps. They examine your clavicles for fractures or irregularities. Your spine gets scrutinized
        for alignment issues, compression fractures (where vertebrae collapse), or destructive lesions that might indicate
        cancer spread to the bones.

        This part is called "disability" because significant bone injuries can limit function and movement. For example,
        multiple rib fractures can make breathing extremely painful and difficult. A spine fracture might risk paralysis
        if the bone fragments press on your spinal cord.

        ### E - Everything Else

        Finally, radiologists review "everything else" - all the small details that don't fit into the other categories
        but can be critically important. They look at the soft tissues of your chest wall and neck for swelling or masses.
        They check for surgical clips, pacemakers, defibrillators, chest tubes, or other medical devices, making sure
        they're properly positioned. They look for air in places it shouldn't be, like under your skin or in the space
        between your lungs.

        Most importantly, they carefully re-examine the areas that are easiest to miss: the very tops of your lungs (the
        apices), the areas behind your heart, the angles where your diaphragm meets your ribs, and the spaces between your
        ribs. These "blind spots" are where small but significant findings - like a small pneumothorax or an early lung
        cancer - might hide if you're not specifically looking for them.
        """)

    st.info("""
    **💡 Why This Matters to You:** When your doctor receives your X-ray report, they're getting the results of this
    thorough, systematic review. Understanding this process can help you appreciate why results might take some time -
    radiologists are being meticulous to ensure nothing is missed. If your doctor wants to order additional imaging
    (like a CT scan), it's not necessarily because something is wrong, but because they want a clearer, more detailed
    view of something they saw on your X-ray.
    """)

    st.markdown("---")

    # AI in Radiology Section
    st.markdown('<div id="ai-in-radiology"></div>', unsafe_allow_html=True)
    st.markdown("## 🤖 Understanding AI in Medical Imaging")

    st.markdown("""
    Artificial intelligence is increasingly being used to help radiologists read medical images, including chest X-rays.
    You might be wondering how a computer can "see" diseases in an X-ray, whether it's accurate, and what role it plays
    in your healthcare. Let's demystify this technology and explain how this particular AI system works.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### How Computers Learn to Read X-Rays

        Teaching a computer to read X-rays is remarkably similar to how humans learn this skill. When medical students
        learn radiology, they spend years looking at thousands of X-rays, with experienced radiologists pointing out what's
        normal and what's abnormal. Over time, their brains learn to recognize patterns - the fuzzy white patches of
        pneumonia, the sharp lines of a collapsed lung, the enlarged silhouette of an overworked heart.

        AI systems learn the exact same way, but much faster. This particular system was trained on 112,120 chest X-rays
        from real patients, each labeled with the diseases visible in that image. The AI "studied" these examples millions
        of times, gradually learning to recognize the visual patterns associated with 14 different chest conditions. Think
        of it like learning to identify birds - at first, all small brown birds look the same, but after seeing hundreds
        of examples, you start noticing subtle differences in beak shape, wing patterns, and behavior that distinguish a
        sparrow from a finch.

        The technical term for this technology is "deep learning using convolutional neural networks" - but don't let the
        jargon intimidate you. A neural network is simply a computer program modeled loosely on how brain cells (neurons)
        work together to process information. The "convolutional" part means it's specifically designed to understand images
        by looking at small pieces of the image (like individual pixels and small patches) and gradually building up to
        understanding larger features (like organ shapes and disease patterns).

        One clever aspect of this system is something called "transfer learning." Instead of learning everything from
        scratch, the AI first learned to recognize general visual features (edges, textures, shapes) by studying over a
        million regular photographs of everyday objects like cars, cats, and buildings. Only then did it specialize in
        medical images. This is similar to how learning to draw generally makes you better at drawing specific things -
        the foundational skills transfer to new applications.

        ### What the AI Can Detect

        This system can look for 14 different conditions simultaneously in a single X-ray: atelectasis (lung collapse),
        cardiomegaly (enlarged heart), consolidation (areas of lung filled with fluid or pus), edema (lung fluid buildup),
        pleural effusion (fluid around the lungs), emphysema (lung damage), fibrosis (lung scarring), hernias, infiltration
        (abnormal lung density), masses (large suspicious areas), nodules (small suspicious areas), pleural thickening,
        pneumonia, and pneumothorax (collapsed lung with air leak).

        For each condition, the AI produces a probability score between 0% and 100% indicating its confidence that the
        condition is present. A score of 75% doesn't mean you're "75% sick" - it means the AI is 75% confident it sees
        signs of that condition based on the patterns it learned during training.
        """)

    with col2:
        st.markdown("""
        ### How Well Does It Work?

        The million-dollar question is: how accurate is this AI? Medical AI systems are measured using something called
        AUC-ROC scores, which range from 0.5 (no better than random guessing) to 1.0 (perfect prediction). This system
        scores between 0.75 and 0.84 depending on which disease it's looking for - that's actually quite good, approaching
        the performance of experienced radiologists for common conditions.

        But here's an important nuance: the AI performs better on some conditions than others. It's particularly good at
        spotting obvious cases - a massive lung consolidation from severe pneumonia, a grossly enlarged heart, or a large
        pleural effusion. It can struggle with subtle findings that even human radiologists debate, like minimal lung
        scarring or tiny nodules partially hidden behind ribs or blood vessels.

        The AI was validated by having expert radiologists review a subset of images and label them independently. In areas
        where the AI and the radiologists agreed, we can be fairly confident. But in areas where they disagreed, it highlights
        that chest X-ray interpretation isn't always black and white - there's genuine medical judgment involved.

        ### What the AI Can and Can't Do

        Let's be clear about where AI excels and where it falls short. The AI is incredibly consistent - it will analyze
        the 1st X-ray of the day exactly the same way as the 1,000th. It never gets tired, distracted, or has an off day.
        It processes each image in seconds, which could help radiologists prioritize urgent cases. In places with a shortage
        of radiologists, it might provide a "first-pass" screening, flagging concerning X-rays that need immediate expert
        review.

        However, the AI has significant limitations. It only sees the X-ray image - it doesn't know your symptoms, your
        medical history, your physical exam findings, or your laboratory test results. A good radiologist integrates all
        of that information. If you came in with chest pain, that changes how they interpret your X-ray compared to if you
        came in with a cough. The AI doesn't know the difference.

        The AI also can't explain its reasoning like a human can. When a radiologist spots something unusual, they can
        explain why it's concerning, compare it to your previous X-rays, and discuss what additional tests might help.
        The AI just outputs numbers. While there are visualization techniques (like Grad-CAM heatmaps that show which parts
        of the image influenced the AI's decision), these are imperfect approximations of the AI's actual "thought process."

        Perhaps most importantly, the AI can only recognize patterns it was trained on. If a new disease emerges (like
        COVID-19 did in 2020) or a rare condition appears, the AI might miss it entirely or misclassify it as something
        it has seen before. Human radiologists can reason by analogy, recognize that something looks unusual even if they've
        never seen it before, and look up similar cases. AI systems lack this flexibility.

        ### The Future Role of AI in Healthcare

        The goal isn't to replace radiologists with AI - that would be dangerous and impractical. Instead, think of AI as
        a powerful assistant. Radiologists currently read dozens or hundreds of images per day, and studies show they
        occasionally miss things, especially subtle findings or when they're fatigued. AI could serve as a tireless second
        set of eyes, flagging potential issues for human review.

        In the future, you might have your X-ray instantly analyzed by AI to flag urgent concerns, but a radiologist would
        still review the image, consider your clinical context, and write the final report your doctor receives. The AI
        helps, but human judgment prevails. This is exactly how this dashboard works - it's an educational tool showing
        what AI can do, not a replacement for medical diagnosis.
        """)

    st.info("""
    **💡 Bottom Line:** AI is a powerful tool that's getting better every year, but it's exactly that - a tool. Your
    medical care should always involve human doctors who can see the full picture of your health, explain their reasoning,
    and make decisions in partnership with you. This AI dashboard demonstrates the technology's capabilities and limitations,
    helping you understand how these tools might assist in your future healthcare.
    """)

    # Back button at bottom
    st.markdown("---")
    if st.button("← Back to Dashboard", type="primary", key="back_bottom"):
        st.session_state.view = 'dashboard'
        st.query_params['view'] = 'dashboard'
        st.rerun()


def render_pathology_section(figures_dir: pathlib.Path):
    """Render the pathology guide section with disease information."""

    st.markdown("""
    Below are the 14 thoracic conditions that this AI system can detect. Each section explains what the condition is,
    how it looks on an X-ray, and why it matters to your health. Reference images for all conditions can be seen in
    the sidebar navigation above.
    """)

    st.markdown("---")

    # Atelectasis
    st.markdown("### 1. Atelectasis")

    col_text, col_img = st.columns([2, 1])

    with col_text:
        st.markdown("""
        **What is it?**

        Atelectasis (pronounced "at-uh-LEK-tuh-sis") is what happens when part of your lung deflates like a balloon losing air.
        Your lungs are made up of millions of tiny air sacs called alveoli, which inflate and deflate with every breath. When
        these air sacs collapse and lose their air, that portion of lung tissue shrinks and becomes denser. Think of it like
        a sponge - when it's full of air it's light and fluffy, but when you squeeze the air out, it becomes smaller and denser.

        **How does it look on an X-ray?**

        On a chest X-ray, a collapsed area of lung appears whiter (more opaque) than healthy lung because the collapsed tissue
        is denser and blocks more X-rays. You'll also notice that the affected area looks smaller - the collapsed lung tissue
        takes up less space. This can cause nearby structures to shift toward the collapsed area, like how nature abhors a vacuum.
        You might see the trachea (windpipe) deviating toward the affected side, or the diaphragm (the breathing muscle below your
        lungs) pulling upward on that side.

        **Why does this happen and why does it matter?**

        Atelectasis is incredibly common after surgery, especially chest or abdominal operations. When you're under anesthesia or
        in pain after surgery, you tend to take shallow breaths and don't cough effectively. This allows mucus to plug up your
        airways, and the lung tissue beyond the blockage collapses. It can also happen if a tumor or foreign object blocks an
        airway, or if you're bedridden for a long time and can't change positions to keep your lungs fully expanded.

        While small areas of atelectasis aren't dangerous and often resolve with deep breathing exercises and coughing, larger
        areas can reduce your oxygen levels and create a breeding ground for pneumonia. That's why after surgery, nurses
        constantly remind you to use that spirometer device and take deep breaths - it's specifically to prevent atelectasis.
        """)

    with col_img:
        render_clickable_xray("Atelectasis", use_thumbnail=True, caption="Atelectasis")

    st.markdown("---")

    # Cardiomegaly
    st.markdown("### 2. Cardiomegaly")

    col_text, col_img = st.columns([2, 1])

    with col_text:
        st.markdown("""
        **What is it?**

        Cardiomegaly (pronounced "car-dee-oh-MEG-uh-lee") literally means "large heart." Your heart is a muscle, and like any
        muscle, it can grow bigger if it has to work harder. But unlike your bicep getting bigger from lifting weights - which
        is beneficial - an enlarged heart usually signals that something is wrong. When your heart has to pump against high blood
        pressure, push blood through damaged valves, or compensate for muscle damage from a heart attack, its chambers enlarge
        and its walls thicken in an attempt to maintain adequate blood flow.

        **How does it look on an X-ray?**

        Diagnosing cardiomegaly on a chest X-ray is straightforward using a measurement called the cardiothoracic ratio. Doctors
        draw a line across your chest at its widest point, then draw another line across your heart at its widest point. In a
        healthy adult, your heart should be less than half the width of your chest. If the heart line is more than 50% of the chest
        line, that's cardiomegaly. The heart appears as an enlarged white shadow in the center-left of your chest, and depending on
        which chambers are enlarged, it might bulge in different directions.

        **Why does this happen and why does it matter?**

        The most common causes of an enlarged heart are high blood pressure (forcing your heart to pump harder against resistance),
        heart valve problems (making your heart work harder to push blood through narrowed valves or compensate for leaky ones),
        and heart failure (where the heart muscle weakens and the chambers dilate trying to pump enough blood). Other causes include
        cardiomyopathy (diseases of the heart muscle itself), chronic lung disease (which strains the right side of your heart), and
        sometimes excessive alcohol use or certain infections.

        Cardiomegaly matters because it's often a sign of heart disease that needs treatment. An enlarged heart works inefficiently -
        it's like trying to pump water with a stretched-out balloon instead of a tight, muscular one. Over time, this can lead to heart
        failure, where your heart can't pump enough blood to meet your body's needs, causing fatigue, shortness of breath, and fluid
        buildup in your lungs and legs. Early detection allows doctors to treat the underlying cause before permanent damage occurs.
        """)

    with col_img:
        render_clickable_xray("Cardiomegaly", use_thumbnail=True, caption="Cardiomegaly")

    st.markdown("---")

    # Consolidation
    st.markdown("### 3. Consolidation")

    col_text, col_img = st.columns([2, 1])

    with col_text:
        st.markdown("""
    **What is it?**

    Consolidation is what doctors call it when the normally air-filled spaces in your lungs get filled with something else - usually
    fluid, pus, blood, or inflammatory cells. Imagine your lungs as a sponge that's supposed to be mostly air. In consolidation,
    portions of that sponge become waterlogged and heavy instead of light and airy. The medical term comes from the word "solid" -
    the affected lung tissue becomes more solid instead of being filled with air.

    **How does it look on an X-ray?**

    Consolidated lung appears as white (opaque) patches on the X-ray because the fluid or pus blocks X-rays just like solid tissue does.
    One characteristic feature is something called "air bronchograms" - you can see dark (air-filled) bronchi (airways) standing out
    against the white consolidated lung tissue, like dark tree branches against a white sky. This happens because the airways themselves
    remain air-filled even though the surrounding lung tissue is filled with fluid. Unlike atelectasis where the lung shrinks,
    consolidation doesn't cause volume loss - the affected area stays the same size, just denser.

    **Why does this happen and why does it matter?**

    The most common cause of consolidation is pneumonia - an infection where bacteria or viruses cause your immune system to flood the
    air sacs with white blood cells and fluid to fight the invaders. But consolidation can also happen from pulmonary edema (fluid
    backing up into your lungs from heart failure), bleeding into the lung (from trauma or blood vessel problems), or aspiration
    (inhaling food, liquids, or stomach contents into your lungs).

    Consolidation matters because it prevents the affected area of lung from doing its job - exchanging oxygen and carbon dioxide.
    Large areas of consolidation can make you seriously short of breath and lower your blood oxygen levels. Bacterial pneumonia with
    consolidation requires antibiotics, while other causes need different treatments. The good news is that once the underlying cause
    is treated, consolidation usually resolves completely and your lung tissue returns to normal.
    """)

    with col_img:
        render_clickable_xray("Consolidation", use_thumbnail=True, caption="Consolidation")

    st.markdown("---")

    # Edema
    st.markdown("### 4. Edema (Pulmonary Edema)")

    col_text, col_img = st.columns([2, 1])

    with col_text:
        st.markdown("""
    **What is it?**

    Pulmonary edema means fluid accumulation in your lungs, and it's one of the most serious findings on a chest X-ray. Think of your
    lungs like a sponge that normally stays dry while air passes through it. In pulmonary edema, that sponge becomes waterlogged,
    making it harder and harder to breathe. The fluid seeps into the air sacs (alveoli) and the spaces between them, effectively
    drowning your lungs from the inside. People with severe pulmonary edema literally feel like they're drowning and can't get enough air.

    **How does it look on an X-ray?**

    Pulmonary edema creates a distinctive appearance on chest X-rays. The classic pattern is called "bat-wing" or "butterfly" opacities -
    bilateral white patches spreading out from the center of your chest like wings. The fluid accumulates first around the blood vessels
    near your heart (the perihilar region) and then spreads outward. You'll also see fine horizontal white lines at the bases of the lungs
    called Kerley B lines - these are thickened connective tissue septa engorged with fluid. In cardiogenic pulmonary edema (caused by heart
    failure), the heart often looks enlarged as well.

    **Why does this happen and why does it matter?**

    There are two main types of pulmonary edema. Cardiogenic pulmonary edema happens when your heart isn't pumping effectively (heart failure),
    causing blood to back up into the vessels of your lungs. The increased pressure forces fluid out of the blood vessels into the lung tissue,
    like water being squeezed out of a sponge. Non-cardiogenic pulmonary edema occurs when the blood vessels in your lungs become leaky due to
    infection, inflammation, toxins, or major trauma - this is often called ARDS (Acute Respiratory Distress Syndrome).

    Pulmonary edema is a medical emergency. It severely impairs your ability to absorb oxygen and can be life-threatening. People with acute
    pulmonary edema often require hospitalization, supplemental oxygen or ventilator support, diuretics (medications to remove excess fluid),
    and treatment of the underlying cause. The good news is that with prompt treatment, pulmonary edema can often be reversed, though the
    underlying condition (like heart failure) typically requires ongoing management.
    """)

    with col_img:
        render_clickable_xray("Edema", use_thumbnail=True, caption="Edema")

    st.markdown("---")

    # Effusion
    st.markdown("### 5. Effusion (Pleural Effusion)")

    col_text, col_img = st.columns([2, 1])

    with col_text:
        st.markdown("""
    **What is it?**

    A pleural effusion is a collection of fluid in the pleural space - the thin gap between your lung and your chest wall. Normally, this
    space contains just a few milliliters of lubricating fluid, like the oil between two engine parts, allowing your lungs to slide smoothly
    against your chest wall as you breathe. In a pleural effusion, excess fluid accumulates in this space - anywhere from a few tablespoons
    to several liters in severe cases. This fluid pushes on your lung from the outside, compressing it and preventing it from fully expanding.

    **How does it look on an X-ray?**

    Pleural effusions have characteristic appearances that change with the amount of fluid. Small effusions show up as blunting of the
    costophrenic angle - that sharp corner where your diaphragm meets your ribs becomes rounded instead of pointed. Moderate effusions create
    a meniscus sign - a curved white line tracking up the side of your chest like water in a glass. Large effusions cause complete
    opacification (whitening) of one side of the chest, obscuring the lung and sometimes pushing the heart and trachea toward the opposite side.
    Gravity pulls the fluid to the bottom of your chest when you're upright, so effusions typically appear at the lung bases.

    **Why does this happen and why does it matter?**

    Pleural effusions are classified as transudates or exudates based on what caused them. Transudates are thin, watery fluid that leaks into
    the pleural space because of pressure imbalances - most commonly from heart failure (the heart can't pump well, causing fluid backup) or
    liver disease. Exudates are thicker, protein-rich fluid that accumulates because of inflammation, infection, or other problems directly
    affecting the pleura - common causes include pneumonia, cancer, blood clots in the lungs, and tuberculosis.

    Small pleural effusions may cause no symptoms, but larger ones can make breathing difficult because they compress the lung. Treatment depends
    on the cause - heart failure effusions are treated with diuretics, infected effusions need antibiotics and drainage, cancerous effusions may
    need to be drained and treated with chemotherapy. Doctors often insert a needle to sample the fluid (thoracentesis) to determine what's
    causing it. Some effusions resolve with treatment of the underlying condition, while others recur and need repeated drainage or more invasive
    procedures.
    """)

    with col_img:
        render_clickable_xray("Effusion", use_thumbnail=True, caption="Effusion")

    st.markdown("---")

    # Emphysema
    st.markdown("### 6. Emphysema")

    col_text, col_img = st.columns([2, 1])

    with col_text:
        st.markdown("""
    **What is it?**

    Emphysema is a type of chronic obstructive pulmonary disease (COPD) where the tiny air sacs in your lungs (alveoli) are permanently damaged
    and destroyed. Imagine a fresh sponge with millions of tiny, separate compartments - that's a healthy lung. Now imagine that sponge melting,
    with the walls between compartments breaking down to form large, useless cavities. That's emphysema. Your lungs can still take in air, but
    they lose their elasticity and can't efficiently push air back out, trapping stale air inside. The total surface area available for oxygen
    exchange drastically decreases.

    **How does it look on an X-ray?**

    Emphysematous lungs look characteristically overinflated and dark on X-rays. The lungs appear larger than normal (hyperinflation) because
    they're trapping air. The diaphragm gets pushed down and flattened instead of maintaining its normal dome shape - doctors count rib spaces,
    and seeing more than 6 ribs above the diaphragm suggests hyperinflation. The lungs look too dark (hyperlucent) because there's more air and
    less tissue than normal. Blood vessel markings are decreased, especially at the edges of the lungs. Sometimes you can see bullae - large air
    pockets where lung tissue has been destroyed, appearing as round dark areas usually at the tops of the lungs.

    **Why does this happen and why does it matter?**

    The overwhelming majority of emphysema cases are caused by cigarette smoking. The toxic chemicals in smoke trigger chronic inflammation in the
    lungs and activate enzymes that digest the elastic walls of the alveoli. After years or decades of smoking, this damage accumulates until it
    becomes symptomatic. A small percentage of emphysema cases are caused by alpha-1 antitrypsin deficiency, a genetic condition where people lack
    a protein that protects lung tissue from these digestive enzymes - these people can develop severe emphysema even in their 30s or 40s without
    smoking.

    Emphysema is irreversible - once those air sac walls are destroyed, they don't grow back. People with emphysema progressively become more
    short of breath, first with exertion and eventually even at rest. They often develop a barrel chest from chronic hyperinflation, and in severe
    cases, they may need supplemental oxygen 24/7. The best treatment is prevention - quitting smoking stops the progression but doesn't reverse
    existing damage. Medications can help open airways and reduce inflammation, pulmonary rehabilitation can improve exercise tolerance, and in
    severe cases, lung volume reduction surgery or lung transplantation might be options.
    """)

    with col_img:
        render_clickable_xray("Emphysema", use_thumbnail=True, caption="Emphysema")

    st.markdown("---")

    # Fibrosis
    st.markdown("### 7. Fibrosis (Pulmonary Fibrosis)")

    col_text, col_img = st.columns([2, 1])

    with col_text:
        st.markdown("""
    **What is it?**

    Pulmonary fibrosis is scarring of your lung tissue. Think about how a scar forms on your skin after a deep cut - the smooth, flexible skin
    is replaced with tough, inflexible scar tissue. The same thing happens in pulmonary fibrosis, except it's happening throughout your lungs.
    The delicate, elastic air sacs and the thin walls where oxygen exchange occurs become thick, stiff, and scarred. Your lungs lose their ability
    to expand easily and become less efficient at transferring oxygen into your bloodstream.

    **How does it look on an X-ray?**

    Fibrotic lungs show a characteristic reticular (net-like) pattern - think of a fishnet stocking stretched over the lung. These are the thickened,
    scarred walls of the air sacs creating a crisscrossing pattern. In advanced cases, you see honeycombing - clustered small cystic air spaces that
    look like a honeycomb, representing destroyed lung architecture. The lungs often appear smaller (volume loss) because the scarred tissue
    contracts. You might also see traction bronchiectasis, where the scarring pulls on airways, making them appear distorted and dilated.

    **Why does this happen and why does it matter?**

    Sometimes pulmonary fibrosis has an identifiable cause: certain medications (like some chemotherapy drugs or heart medications), radiation therapy
    to the chest, occupational exposures (like asbestos, silica dust, or metal dusts), autoimmune diseases (like rheumatoid arthritis or scleroderma),
    or chronic hypersensitivity reactions (like from bird droppings or moldy hay). But frustratingly, in more than half of cases, doctors never find
    a cause - this is called idiopathic pulmonary fibrosis (IPF), meaning "lung scarring of unknown origin."

    Pulmonary fibrosis is a serious, progressive disease. Unlike inflammation which can improve with treatment, scar tissue is permanent. People with
    pulmonary fibrosis experience progressive shortness of breath, first with exertion and eventually at rest. They often have a chronic dry cough.
    Oxygen levels drop, especially during physical activity or sleep. Treatment focuses on slowing progression (with medications like pirfenidone or
    nintedanib for IPF), managing symptoms, supplemental oxygen, and pulmonary rehabilitation. For younger patients with advanced disease, lung
    transplantation may be an option. Unfortunately, the prognosis for IPF is poor, with median survival of 3-5 years from diagnosis, though this
    varies widely and newer medications offer hope for slowing progression.
    """)

    with col_img:
        render_clickable_xray("Fibrosis", use_thumbnail=True, caption="Fibrosis")

    st.markdown("---")

    # Hernia
    st.markdown("### 8. Hernia (Hiatal Hernia)")

    col_text, col_img = st.columns([2, 1])

    with col_text:
        st.markdown("""
    **What is it?**

    When we talk about hernias on chest X-rays, we're usually referring to hiatal hernias, where part of your stomach pushes up through your
    diaphragm into your chest cavity. Your diaphragm is the large, dome-shaped muscle that separates your chest from your abdomen and powers
    your breathing. There's an opening (hiatus) in the diaphragm where your esophagus (food pipe) passes through to connect to your stomach.
    In a hiatal hernia, this opening enlarges, allowing your stomach to slide up into your chest, especially when you lie down or bend over.

    **How does it look on an X-ray?**

    A hiatal hernia appears as an abnormal air or air-fluid level behind your heart (in the retrocardiac space). You might see a bubble of air or
    a horizontal line indicating an air-fluid level in an unexpected location - that's part of your stomach sitting in your chest instead of in
    your abdomen. Sometimes you can see the actual outline of the herniated stomach pouch. In traumatic diaphragmatic hernias (much less common,
    usually from severe car accidents or falls), you might see loops of bowel - recognizable as air-filled, tubular structures - extending up
    into the chest.

    **Why does this happen and why does it matter?**

    Hiatal hernias are extremely common, especially in people over 50, and most people don't even know they have one. They're often found
    incidentally on chest X-rays done for other reasons. Risk factors include obesity (increased abdominal pressure pushing organs upward), aging
    (the diaphragm weakens over time), persistent coughing or straining, and sometimes trauma. Many hiatal hernias cause no symptoms at all.

    When symptoms do occur, they're usually related to acid reflux - the stomach sliding up into the chest disrupts the normal anti-reflux
    mechanisms, allowing stomach acid to flow back into the esophagus, causing heartburn, chest pain, difficulty swallowing, or a chronic cough.
    Large hiatal hernias can occasionally cause breathing difficulties or chest pain by compressing other structures in the chest.

    Most hiatal hernias don't require specific treatment beyond managing reflux symptoms with lifestyle changes (elevating the head of your bed,
    avoiding large meals before bedtime, losing weight) and medications (antacids, H2 blockers, or proton pump inhibitors). Surgery is reserved for
    large hernias causing severe symptoms, hernias that don't respond to medical management, or complications like the stomach becoming twisted or
    trapped in the chest. Traumatic diaphragmatic hernias, by contrast, are surgical emergencies requiring repair to prevent bowel strangulation.
    """)

    with col_img:
        render_clickable_xray("Hernia", use_thumbnail=True, caption="Hernia")

    st.markdown("---")

    # Infiltration
    st.markdown("### 9. Infiltration")

    col_text, col_img = st.columns([2, 1])

    with col_text:
        st.markdown("""
    **What is it?**

    Infiltration is a non-specific term radiologists use to describe haziness or cloudiness in the lungs on an X-ray. It's not a specific disease -
    rather, it's a descriptive term meaning something is filling the air spaces or interstitium (the supporting tissue between air sacs) that
    shouldn't be there. Think of looking through a clean window versus a foggy one - infiltration is like fog in your lungs, making them appear
    less dark (less air-filled) on X-rays. The term is somewhat vague on purpose because the same hazy appearance can be caused by many different
    conditions.

    **How does it look on an X-ray?**

    Infiltrates appear as ill-defined, hazy opacities - areas that look whiter than normal lung but not as sharply defined as consolidation. The
    edges are fuzzy and gradual rather than sharp. They can be patchy (scattered irregular areas), diffuse (widespread throughout the lungs), or
    localized to one region. Unlike consolidation, you usually don't see air bronchograms (visible airways) within infiltrates. The appearance has
    been described as looking like ground glass - you can still make out the underlying structures but everything looks hazy.

    **Why does this happen and why does it matter?**

    The challenge with infiltrates is that the differential diagnosis is huge - many conditions can cause this appearance. Early-stage pneumonia
    often starts as infiltration before progressing to consolidation. Pulmonary edema (fluid in the lungs from heart failure) can appear as
    infiltrates. Hypersensitivity pneumonitis (allergic lung inflammation from inhaled organic dusts) causes diffuse infiltrates. Certain drug
    reactions, viral infections, early-stage lung injury from various causes, and even some types of lung cancer can present as infiltrates.

    Because "infiltration" is non-specific, doctors never stop with this finding. They use your clinical history (Are you short of breath? Do you
    have a fever? Have you been around sick people?), physical exam, and other tests to narrow down the cause. Treatment obviously depends on
    identifying the underlying condition - antibiotics for bacterial pneumonia, stopping an offending drug, treating heart failure, etc. Sometimes
    infiltrates represent something trivial that resolves on its own; other times they're the first sign of serious disease. Follow-up X-rays to
    see if infiltrates are improving, staying stable, or worsening provide important clues to the diagnosis.
    """)

    with col_img:
        render_clickable_xray("Infiltration", use_thumbnail=True, caption="Infiltration")

    st.markdown("---")

    # Mass
    st.markdown("### 10. Mass")

    col_text, col_img = st.columns([2, 1])

    with col_text:
        st.markdown("""
    **What is it?**

    In radiology terms, a mass is any focal (localized) opacity in the lung that measures more than 3 centimeters (about 1.2 inches) in diameter.
    Anything smaller than 3 cm is called a nodule rather than a mass. While the size cutoff is arbitrary, masses are more concerning than nodules
    because they're more likely to be malignant (cancerous). A mass represents a concentrated area of abnormal tissue - it could be a tumor (benign
    or malignant), an abscess (a pocket of infection), a large area of organizing pneumonia, or more rarely, something else entirely.

    **How does it look on an X-ray?**

    A mass appears as a round, oval, or irregular white (opaque) area within the normally dark lung tissue. The margins (edges) of the mass provide
    important clues. Smooth, well-defined borders often suggest a benign process like a granuloma (a ball of inflammatory cells walling off an old
    infection) or a benign tumor. Irregular, spiculated borders (with spiky projections extending outward, sometimes described as "corona radiata"
    or sun-ray appearance) strongly suggest malignancy - these are the fingers of cancerous tissue invading into the surrounding lung. Masses can
    occur anywhere in the lung, and their location sometimes provides clues about the cause.

    **Why does this happen and why does it matter?**

    Unfortunately, a lung mass raises immediate concern for cancer, particularly if you're a current or former smoker or older than 50. Primary lung
    cancer (cancer originating in the lung) and metastatic disease (cancer that spread to the lungs from elsewhere in the body) are common causes of
    lung masses. However, not all masses are malignant. Large granulomas (from old tuberculosis or fungal infections), lung abscesses (infections
    that have formed a walled-off pocket of pus), organizing pneumonia, and benign tumors can also present as masses.

    Finding a lung mass triggers an immediate workup. Doctors will compare the current X-ray with any previous imaging - masses that have been stable
    for years are less concerning than new masses or ones that are growing. Almost always, the next step is a CT scan, which provides much more detail
    than a chest X-ray and can better characterize the mass. Depending on the CT findings, further steps might include a PET scan (to see if the mass
    is metabolically active like cancer typically is), biopsy (either through the skin, through a bronchoscope down your airways, or surgically),
    and staging tests if cancer is confirmed.

    The prognosis depends entirely on what the mass turns out to be. Benign masses might just need monitoring or surgical removal if they're causing
    symptoms. Malignant masses require oncology consultation, staging, and treatment which might include surgery, radiation, chemotherapy,
    immunotherapy, or combinations thereof. Early-stage lung cancers confined to a small mass have much better outcomes than advanced disease, which
    is why lung cancer screening with low-dose CT scans is now recommended for high-risk individuals.
    """)

    with col_img:
        render_clickable_xray("Mass", use_thumbnail=True, caption="Mass")

    st.markdown("---")

    # Nodule
    st.markdown("### 11. Nodule")

    col_text, col_img = st.columns([2, 1])

    with col_text:
        st.markdown("""
    **What is it?**

    A pulmonary nodule is a small, round or oval spot in the lung that measures less than 3 centimeters (about 1.2 inches) in diameter. Anything
    larger is called a mass instead of a nodule. Nodules are incredibly common - they're found in about 1 out of every 4 chest CTs (though less
    frequently visible on regular chest X-rays because they can be too small or hidden behind other structures). The vast majority of pulmonary
    nodules turn out to be benign (non-cancerous), but some are early lung cancers, which is why they always need careful evaluation.

    **How does it look on an X-ray?**

    A nodule appears as a small, round or oval white spot within the dark lung fields. Nodules can be solid (completely white), ground-glass (hazy,
    semi-transparent), or part-solid (a mix of both). Like masses, the borders matter - smooth, well-defined edges favor benign causes, while
    irregular or spiculated (spiky) edges raise concern for cancer. Calcification (calcium deposits) within a nodule often indicates it's benign -
    patterns like popcorn calcification suggest a hamartoma (a benign, disorganized growth of normal lung tissue), while central or laminated
    calcification suggests an old granuloma from a healed infection.

    **Why does this happen and why does it matter?**

    The list of causes for pulmonary nodules is long. Benign causes include granulomas (scars from old infections like tuberculosis, histoplasmosis,
    or coccidioidomycosis), hamartomas (benign tumors made of cartilage and other normal tissues in abnormal arrangements), inflammatory nodules
    (from conditions like rheumatoid arthritis or sarcoidosis), intrapulmonary lymph nodes, and arteriovenous malformations. Malignant causes include
    primary lung cancer (especially concerning in current or former smokers) and metastatic cancer (spread from cancers elsewhere in the body -
    especially kidney, colon, breast, or melanoma).

    The probability that any individual nodule is malignant varies widely based on risk factors. In a 70-year-old smoker with a 2 cm spiculated nodule,
    the chance of cancer might be 60-70%. In a 30-year-old non-smoker with a small, smooth, calcified 4 mm nodule, the chance might be less than 1%.

    How nodules are managed depends on their size, appearance, and your risk factors. Very small nodules (under 6 mm) in low-risk people might just
    need follow-up imaging in 6-12 months to ensure they're not growing. Larger nodules or concerning features often lead to CT scans for better
    characterization, PET scans to assess metabolic activity, and sometimes biopsy or surgical removal. Many nodules end up being monitored with
    serial imaging - stability over 2 years strongly suggests the nodule is benign. The discovery of a lung nodule is anxiety-provoking, but remember
    that the majority turn out to be harmless old scars or benign growths.
    """)

    with col_img:
        render_clickable_xray("Nodule", use_thumbnail=True, caption="Nodule")

    st.markdown("---")

    # Pleural Thickening
    st.markdown("### 12. Pleural Thickening")

    col_text, col_img = st.columns([2, 1])

    with col_text:
        st.markdown("""
    **What is it?**

    Pleural thickening is exactly what it sounds like - the pleura (the thin membrane that lines your lungs and chest wall) becomes thicker than
    normal. Normally, the pleura is so thin you can't even see it on a chest X-ray. When it thickens due to inflammation, scarring, or tumor
    infiltration, it becomes visible as a white line or sheet along the edge of the lung. Think of it like the lining of a jacket getting stiff
    and thick instead of remaining thin and flexible.

    **How does it look on an X-ray?**

    Pleural thickening appears as increased density (whiteness) along the chest wall, following the contour of the ribs. It can be focal (affecting
    just a small area) or diffuse (spreading along much of the pleural surface). Unlike a pleural effusion (fluid collection) which shifts when you
    change position, pleural thickening is fixed and doesn't move. It can be smooth and uniform, or irregular and nodular depending on the cause.
    Sometimes it's associated with calcification - calcium deposits in the thickened pleura that appear as very bright white on X-rays.

    **Why does this happen and why does it matter?**

    The most important cause of pleural thickening to recognize is asbestos exposure. People who worked in construction, shipyards, or other
    industries where asbestos was used (particularly before the 1980s) often develop pleural plaques - areas of focal pleural thickening that can
    become calcified. While pleural plaques themselves are benign and don't usually cause symptoms, they mark someone as having had asbestos exposure,
    which increases their risk for mesothelioma (a rare but deadly cancer of the pleura) and lung cancer.

    Other causes of pleural thickening include previous infections (like tuberculosis or empyema - pus in the pleural space), previous hemothorax
    (blood in the pleural space from trauma), inflammatory conditions (like rheumatoid arthritis or lupus), radiation therapy to the chest, and
    previous pleural effusions. In some cases, irregular, nodular pleural thickening can be caused by mesothelioma itself.

    Most pleural thickening is discovered incidentally and doesn't cause symptoms or require treatment. Extensive pleural thickening can occasionally
    restrict lung expansion and cause breathing difficulties, but this is uncommon. The main clinical significance is recognizing asbestos-related
    changes (which warrant lung cancer screening and patient education) and distinguishing benign pleural thickening from mesothelioma (which requires
    biopsy if there's concern). Once formed, pleural thickening is generally permanent - the scarring doesn't resolve.
    """)

    with col_img:
        render_clickable_xray("Pleural_Thickening", use_thumbnail=True, caption="Pleural Thickening")

    st.markdown("---")

    # Pneumonia
    st.markdown("### 13. Pneumonia")

    col_text, col_img = st.columns([2, 1])

    with col_text:
        st.markdown("""
    **What is it?**

    Pneumonia is an infection of the lung tissue itself - specifically, the air sacs (alveoli) and the small airways. When bacteria, viruses, or
    fungi invade your lungs, your immune system responds by sending white blood cells and flooding the area with fluid to fight the infection.
    This inflammatory response fills the normally air-filled alveoli with fluid and debris, preventing oxygen from reaching your bloodstream
    efficiently. Pneumonia remains one of the leading infectious causes of death worldwide, though most cases in healthy people respond well to
    antibiotics if bacterial, or resolve on their own if viral.

    **How does it look on an X-ray?**

    Pneumonia typically appears as consolidation - a white (opaque) area representing fluid-filled lung tissue. One characteristic feature is air
    bronchograms - the dark outlines of air-filled bronchi (airways) visible against the white consolidated lung, like tree branches against a cloudy
    sky. The consolidation can be focal (confined to one segment or lobe of the lung), multifocal (several scattered patches), or diffuse (widespread).
    Bacterial pneumonia often produces dense, lobar consolidation (an entire lobe of the lung appears white). Viral or atypical pneumonia tends to
    produce a more patchy, interstitial pattern. Sometimes you'll see an associated pleural effusion (fluid collection outside the lung) in the same
    area.

    **Why does this happen and why does it matter?**

    Pneumonia happens when microorganisms overwhelm your lung defenses. Bacteria like Streptococcus pneumoniae (the most common cause of community-
    acquired pneumonia), Haemophilus influenzae, or Staphylococcus aureus can infect healthy lungs, especially after a viral cold weakens your
    defenses. Viruses like influenza, RSV, or COVID-19 can directly cause viral pneumonia. Fungi like Pneumocystis (in immunocompromised people)
    or endemic fungi in certain geographic areas can also cause pneumonia. Aspiration pneumonia occurs when you inhale food, liquids, or vomit into
    your lungs, often during unconsciousness or swallowing difficulties.

    Risk factors include advanced age (weakened immune system), chronic lung diseases (like COPD or cystic fibrosis), smoking (damages the lung's
    cleaning mechanisms), immunosuppression (from HIV, cancer chemotherapy, or medications), and chronic diseases like diabetes or heart failure.

    Symptoms typically include cough (often producing discolored phlegm), fever, chills, chest pain that worsens with breathing, and shortness of
    breath. Elderly people sometimes present atypically with confusion instead of fever. Bacterial pneumonia is treated with antibiotics - the
    specific antibiotic chosen depends on the suspected organism and local resistance patterns. Viral pneumonia is usually supportive care (rest,
    fluids, fever reducers), though antiviral medications exist for influenza and COVID-19. Severe pneumonia requires hospitalization for IV
    antibiotics, oxygen, and close monitoring. Vaccines exist for pneumococcus (the most common bacterial cause) and influenza, significantly
    reducing pneumonia risk in vaccinated individuals.
    """)

    with col_img:
        render_clickable_xray("Pneumonia", use_thumbnail=True, caption="Pneumonia")

    st.markdown("---")

    # Pneumothorax
    st.markdown("### 14. Pneumothorax")

    col_text, col_img = st.columns([2, 1])

    with col_text:
        st.markdown("""
    **What is it?**

    A pneumothorax (pronounced "new-moh-THOR-aks") is air in the pleural space - the normally microscopic gap between your lung and chest wall.
    Your lungs are like balloons inside the chest cavity. Normally, they stay inflated because there's a vacuum seal in the pleural space. When air
    gets into that space (from a hole in the lung or a penetrating chest injury), it breaks the vacuum seal and the lung collapses, just like a
    balloon deflating. The medical term literally means "air-chest" (pneumo = air, thorax = chest).

    **How does it look on an X-ray?**

    The hallmark of pneumothorax is seeing the visceral pleural line - a thin white line outlining the edge of the collapsed lung, with no lung
    markings (blood vessels) visible beyond it. Outside this line, you see abnormal blackness (hyperlucency) - that's the air that shouldn't be
    there, looking darker than even normal lung because it's pure air with no blood vessels. Small pneumothoraces might only be visible at the very
    top (apex) of the lung where air rises naturally. Large pneumothoraces show extensive lung collapse, sometimes with the entire lung compressed
    into a small ball near the center of the chest.

    A tension pneumothorax is a life-threatening emergency where air continues to accumulate under pressure, compressing not just the lung but also
    the heart and major blood vessels. On X-ray, you'll see the trachea and heart shifted away from the pneumothorax side, and the diaphragm may be
    pushed down. This is a clinical diagnosis that shouldn't wait for X-ray - it requires immediate needle decompression.

    **Why does this happen and why does it matter?**

    Pneumothoraces are classified by cause:

    **Spontaneous pneumothorax** occurs without obvious trauma. Primary spontaneous pneumothorax happens in young, tall, thin people (often males in
    their 20s) with no lung disease - small blebs (weak spots) on the lung surface rupture for no apparent reason. Secondary spontaneous pneumothorax
    occurs in people with underlying lung disease like COPD, cystic fibrosis, or severe asthma - diseased lung tissue is more prone to rupture.

    **Traumatic pneumothorax** results from chest injuries - rib fractures puncturing the lung, stab wounds, gunshot wounds, or blunt force trauma
    that tears lung tissue. Car accidents and falls from height are common causes.

    **Iatrogenic pneumothorax** is caused by medical procedures - central line placement (inserting IV catheters into large neck or chest veins),
    lung biopsies, mechanical ventilation (the pressure can rupture lung tissue), or even overly aggressive CPR.

    Small pneumothoraces (less than 15-20% of the lung) in stable patients might just be observed, breathing supplemental oxygen to help the air
    reabsorb faster. Larger pneumothoraces or symptomatic ones require a chest tube - a plastic tube inserted through the ribs to evacuate the air
    and re-expand the lung. The tube stays in place for several days until the leak seals. Some people experience recurrent pneumothoraces - after
    one pneumothorax, there's about a 30% chance of another one, and the risk increases with each recurrence. Recurrent cases might require surgical
    intervention (pleurodesis - intentionally scarring the pleura to prevent reaccumulation, or surgery to remove blebs).

    Untreated pneumothorax can be life-threatening if the lung collapse is extensive enough to impair breathing and oxygen levels. Tension
    pneumothorax is immediately lethal if not recognized and treated emergently. But with appropriate treatment, most pneumothoraces resolve
    completely and the lung fully re-expands.
    """)

    with col_img:
        render_clickable_xray("Pneumothorax", use_thumbnail=True, caption="Pneumothorax")

    st.markdown("---")

    st.info("""
    **💡 Remember:** These descriptions are for educational purposes. If you or someone you care for has been diagnosed with any of these
    conditions, your doctor can provide personalized information about what it means for your specific situation, treatment options, and
    prognosis. Every patient is different, and these general descriptions can't capture the nuances of individual cases.
    """)
