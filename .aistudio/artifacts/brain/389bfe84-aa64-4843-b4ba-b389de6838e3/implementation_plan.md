# Online AI Photo Style (HD) & Zero-Glitch Face Swap — முழுமையான திட்டம் (Implementation Plan)

ஆன்லைன் AI தளங்கள் (Remaker AI, FaceFusion HD, InsightFace Studio) போல **100% தெளிவான, இயற்கையான, HD தரத்திலான (Online Photo Style)** Output வர என்னென்ன `.onnx` மாடல்கள் தேவை மற்றும் ஆப்பின் உள்ளே நாம் செய்யப்போகும் **5 முக்கிய தொழில்நுட்ப மாற்றங்கள்** கீழே பட்டியலிடப்பட்டுள்ளன.

---

### User Review & Critical Decisions

> [!IMPORTANT]
> **ஆன்லைன் Photo Style (HD) தரம் வர தேவையான `.onnx` மாடல்கள்:**
> `inswapper_128.onnx` மாடல் மட்டுமே தனியாக `128×128` என்ற சிறிய அளவில்தான் படத்தை உருவாக்கும். ஆன்லைன் ஆப்களில் படம் HD தரத்தில் வர அவர்கள் **4 மாடல்களை ஒன்றாகப்** பயன்படுத்துகிறார்கள்:
> 1. **`det_10g.onnx` (~16 MB - கட்டாயம்):** முகத்தையும் 5 கண்/மூக்கு/வாய் புள்ளிகளையும் துல்லியமாகக் கண்டறிய.
> 2. **`w600k_r50.onnx` (~166 MB - மிக முக்கியம்):** Source முகத்தின் 100% உண்மையான முகச் சாயலை (512-D ArcFace Identity) பிரித்தெடுக்க. (இது இல்லாமல் 2-Model Mode-ல் இயக்கினால் முழுமையான சாயல் வராது).
> 3. **`inswapper_128.onnx` (~554 MB - கட்டாயம்):** முகத்தை மாற்றியமைக்கும் பிரதான மாடல்.
> 4. **`gfpgan_1.4.onnx` (~340 MB - Online HD Look-க்கு மிக முக்கியம்!):** `inswapper_128` தரும் மங்கலான `128×128` முகத்தை **`512×512` Super-HD (16 மடங்கு அதிக துல்லியம்!)** ஆக மாற்றி, கண்கள், புருவ முடிகள், உதடு மற்றும் சருமத்தை ஆன்லைன் ஃபோட்டோ ஸ்டைலில் பளபளப்பாக மாற்றும் மாடல்!
> 5. **`segformer_B5_ce.onnx` (விருப்பத் தேர்வு - Occlusion Mask):** நெற்றி முடி, கன்னத்தில் விழும் முடி, மூக்குத்தி, பொட்டு ஆகியவற்றைப் பாதுகாத்து இயற்கையாகக் கலக்க.

- **Confirmed Decision 1 (Direct 512×512 HD Blending Canvas — மிகப்பெரிய மாற்றம்!)**:
  தற்போது ஆப்பில் `gfpgan_1.4.onnx` மாடல் `512×512` HD படத்தை உருவாக்கிய பிறகும், அதை மீண்டும் `128×128` ஆகச் சுருக்கி (`downscale`) ஒட்டுகிறது! இதை மாற்றி **நேரடியாக `512×512` High-Resolution அளவிலேயே Target படத்தில் Warp செய்து ஒட்ட உள்ளோம்**. (`gfpgan_1.4.onnx` இல்லாவிட்டாலும் Built-in 512×512 Guided Super-Resolution Enhancer மூலம் HD தரம் கிடைக்கும்).
- **Confirmed Decision 2 (Zero Source-Pixel Ghosting)**:
  Source படத்தின் பிக்சல்களை நேரடியாக கன்னம் அல்லது புருவத்தின் மேல் கலப்பதை முற்றிலும் நீக்கிவிடுவதால், உங்கள் படத்தில் வலது கன்னத்தில் வந்த **கருப்பு முடி கோடு** மற்றும் இடது புருவத்தில் வந்த **வெள்ளை இடைவெளி** 100% நீங்கிவிடும்.
- **Confirmed Decision 3 (Dynamic Landmark Eye & Catchlight Engine)**:
  தலை சாய்ந்திருந்தாலும் `M * targetFace.landmarks5` வழியாக வலது மற்றும் இடது கண்களின் உண்மையான மையத்தைக் கண்டறிந்து, கருவிழி (Iris), வெள்ளை விழி (Sclera) மற்றும் கண்ணின் பளபளப்பை (Corneal Catchlight) ஆன்லைன் ஸ்டுடியோ தரத்தில் மீட்டெடுக்க உள்ளோம்.

---

### 1. Overview & Core Concept

- **What It Does (என்ன செய்யப் போகிறோம்)**:
  ஆஃப்லைனிலேயே ஆன்லைன் AI ஆப்களுக்கு இணையான **512×512 High-Definition Face Swap** உருவாக்குதல் — எந்தவொரு AI Artifacts, கன்னக் கோடுகள், புருவ வெட்டுகள் அல்லது மங்கலான கண்கள் இல்லாமல், இயற்கையான சருமத் துளைகள் (Skin Pores) மற்றும் ஸ்டுடியோ லைட்டிங்குடன் (Studio Relighting) படத்தை உருவாக்கும்.
- **Target Audience / Persona**:
  தொழில்முறை தரத்திலான (Online Studio-Grade) உருவப்படங்கள், பாரம்பரிய உடைப் படங்கள் (Saree/Festive Portraits) மற்றும் சாய்வான கோணப் படங்களில் துல்லியமான Face Swap விரும்பும் பயனர்கள்.
- **Key Value (முக்கிய பலன்)**:
  - **16× அதிக தெளிவு (`512×512` HD Pipeline):** முகம் மங்கலாக இல்லாமல், முழுப் புகைப்படத்தின் HD தரத்தோடு அப்படியே பொருந்தும்.
  - **Zero Visual Glitches:** கன்னத்தில் முடி கோடுகளோ, புருவத்தில் இடைவெளியோ, வலது கண்ணில் வெள்ளை படலமோ வராது.

---

### 2. User Experience & Visual Design

- **Key User Flows (பயன்பாட்டு முறை)**:
  1. **Source & Target தேர்வு:** பயனர் இரு படங்களையும் தேர்ந்தெடுக்கிறார்.
  2. **Online Photo Style HD Pipeline:**
     - `w600k_r50.onnx` + `inswapper_128.onnx` மூலம் முக மாற்றம் நிகழ்கிறது.
     - உடனே முகம் **512×512 HD Canvas**-க்கு உயர்த்தப்பட்டு (`gfpgan_1.4.onnx` அல்லது Built-in 512×512 Detail Restorer மூலம்), சருமத் துளைகள் (Skin Micro-Texture), கண் கருவிழி மற்றும் புருவங்கள் மெருகூட்டப்படுகின்றன.
     - நேரடியாக 512×512 துல்லியத்தில் Target புகைப்படத்தில் ஒட்டப்படுகிறது.
- **Visual Identity & Theme**:
  - *Aesthetic Direction*: High-Craft Dark Studio UI — தெளிவான Before/After ஒப்பீடு மற்றும் HD தரக் காட்டிகள்.
  - *Color Palette & Mood*: Deep Obsidian (`#0B1021`), Electric Cyan (`#00E5FF`), Emerald (`#10B981`).
  - *Typography & Hierarchy*: தெளிவான படிநிலைத் தலைப்புகள் மற்றும் மாடல் நிலைக் காட்டிகள்.
  - *Component Styling & Layout*: Material Design 3 அட்டைகள் மற்றும் உடனடி மாடல் சரிபார்ப்பு.
- **Interactive Feedback & Motion**:
  - 5-கட்ட முன்னேற்றப் பட்டியில் *"512x512 Online-Style HD Restoration & Dynamic Eye/Skin Blending"* நிலை காட்டப்படும்.

---

### 3. Key Product Decisions & Trade-Offs

- **Decision 1: 128×128-க்கு பதிலாக 512×512 Direct HD Blending**
  - *Chosen Approach*: Blending மற்றும் Eye/Skin Restoration அனைத்தையும் 128×128-ல் செய்வதற்குப் பதிலாக **512×512 (அல்லது High-Res) அளவில்** செய்து, நேரடியாக Target படத்தில் Warp செய்தல்.
  - *Why*: `inswapper_128` தரும் குறைந்த தெளிவை (Low-res blur) நீக்கி, ஆன்லைன் ஆப் போன்ற கூர்மையான கண் இமைகள், உதடு மற்றும் சருமத் தெளிவைத் தருகிறது.
  - *Alternatives Considered*: 128×128 அளவிலேயே ஒட்டுவது (இதுவே முகம் மட்டும் மங்கலாகத் தெரியக் காரணம்).

- **Decision 2: High-Frequency Skin Pore & Specular Transfer (ஆன்லைன் Photo Style சரும பளபளப்பு)**
  - *Chosen Approach*: Target முகத்தில் உள்ள இயற்கையான சரும நுணுக்கங்களை (High-Frequency Skin Texture & Highlight Sheen) பிரித்தெடுத்து, மாற்றப்பட்ட முகத்தின் சருமத்தில் மட்டும் மென்மையாகச் சேர்த்தல் (கன்னத்தில் முடி/கோடுகள் இருந்தால் அவற்றை Occlusion Threshold மூலம் தவிர்த்துவிடுதல்).
  - *Why*: AI உருவாக்கிய பிளாஸ்டிக் போன்ற (Plastic/Wax skin) தோற்றத்தை நீக்கி, நிஜக் கேமராவில் எடுத்த புகைப்படம் போன்ற தோற்றத்தைத் தருகிறது.

- **Decision 3: Dynamic Landmark Eye & Eyebrow Protection**
  - *Chosen Approach*: `M * targetFace.landmarks5` வழியாக கண்களின் உண்மையான இருப்பிடத்தைக் கணக்கிட்டு, இரு கண்களின் கருவிழி, வெள்ளை விழி மற்றும் புருவங்களை 100% துல்லியமாகப் பாதுகாத்தல்.
  - *Why*: தலை எந்தக் கோணத்தில் சாய்ந்திருந்தாலும் வலது கண்ணோ புருவமோ சிதையாது.

---

### 4. Technical Architecture & Data Strategy *(Technical Reference)*

- **Architecture & Component Diagram**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  SOURCE PHOTO & TARGET PHOTO INPUTS                     │
└───────────────────┬─────────────────────────────────┬───────────────────┘
                    │                                 │
                    ▼                                 ▼
┌───────────────────────────────────┐ ┌───────────────────────────────────┐
│  det_10g.onnx + w600k_r50.onnx    │ │  Target Dynamic 5-Point Geometry  │
│ (512-D ArcFace + True emap[512²]) │ │ (Exact Eye/Brow/Nose/Mouth Coords)│
└───────────────────┬───────────────┘ └───────────────┬───────────────────┘
                    └───────────────┬─────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              inswapper_128.onnx CORE IDENTITY SYNTHESIS                 │
│        (Zero Source-Pixel Ghosting — Pure Neural Pose Transfer)         │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│         512x512 ONLINE-STYLE HD SUPER-RESOLUTION & RESTORATION          │
│  • Optional gfpgan_1.4.onnx [1,3,512,512] OR Built-in 512x512 Upscaler  │
│  • Dynamic Landmark Right & Left Eye Restorer (Deep Iris & Catchlight)  │
│  • Eyebrow Continuity Guard (No white gaps or double eyebrows)          │
│  • Skin Pore Micro-Texture & LAB Studio Relighting Harmonizer           │
│  • Direct 512x512 -> Full-Res Target Inverse Affine Feather Blending    │
└─────────────────────────────────────────────────────────────────────────┘
```

- **Data Model & State**:
  - **Dynamic Landmarks (`warpedLandmarks`)**: 5 முகப் புள்ளிகளும் Warped Crop இடத்திற்கு மாற்றப்பட்டு கண் மற்றும் புருவ மண்டலங்களைத் துல்லியமாக வழிநடத்தும்.
  - **512×512 HD Synthesis Buffer**: `gfpgan_1.4.onnx` (அல்லது Built-in HD Restorer) உருவாக்கும் `512×512` படம் மீண்டும் `128×128`-ஆக சுருக்கப்படாமல், `4×` பெருக்கப்பட்ட உருமாற்ற மேட்ரிக்ஸ் (`m512`) மூலம் நேரடியாக Target படத்தில் ஒட்டப்படும்.
- **Interactive Component & State Mapping**:
  - பயனர் **`Proceed`** கொடுத்தவுடன், கன்னக் கோடுகள் மற்றும் புருவப் பிழைகளை நீக்கும் **Zero-Ghosting Engine**, **Dynamic Right/Left Eye Restorer**, மற்றும் **Direct 512×512 Online-Style HD Blender** ஆகியவை கோடில் இணைக்கப்பட்டு உடனடியாகச் செயல்படும்.
