# 🎯 User Guide - Disability Job Matching System

**Complete Operational Manual for Employment Center and SIL Operators**

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Getting Started](#getting-started)
3. [Main Interface Tour](#main-interface-tour)
4. [Finding Matches for Candidates](#finding-matches-for-candidates)
5. [Understanding Results](#understanding-results)
6. [Analytics Dashboard](#analytics-dashboard)
7. [Dataset Management](#dataset-management)
8. [Configuration Settings](#configuration-settings)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## 📊 System Overview

### What is the Disability Job Matching System?

This system is an advanced AI-powered tool designed to help Employment Centers (CPI) and Labor Integration Services (SIL) find the best company matches for candidates with disabilities. It automates the complex process of evaluating candidate exclusions against company compatibility requirements.

### Key Benefits for Operators

- **⏱️ Time Savings**: Reduces manual matching from hours to seconds
- **🎯 Improved Accuracy**: 90%+ compatibility scoring vs subjective evaluation
- **📊 Data-Driven Decisions**: Objective scoring based on multiple factors
- **🔍 Comprehensive Search**: Evaluates all companies within specified radius
- **📈 Analytics**: Track placement patterns and system performance

### Who Should Use This Guide?

- **Employment Center Operators** (CPI staff)
- **SIL Coordinators** (Labor Integration Service staff)
- **Case Managers** responsible for candidate placement
- **Account Managers** handling company relationships

---

## 🚀 Getting Started

### System Requirements

- **Web Browser**: Chrome, Firefox, Safari, or Edge (latest versions)
- **Internet Connection**: Required for initial setup and geocoding
- **Screen Resolution**: Minimum 1024x768 (1920x1080 recommended)
- **No Installation Required**: System runs entirely in web browser

### First Time Access

1. **Open the System**:
   ```
   Open your web browser and navigate to the system URL
   (provided by your IT administrator)
   ```

2. **System Initialization**:
   - The system will automatically load with demo data on first run
   - Initial loading may take 30-60 seconds
   - You'll see the main dashboard with sample candidates and companies

3. **Verify System Status**:
   - Check the "Info Sistema" tab to confirm all components are working
   - Ensure you see "✅ Interface: Operativo" status

### Demo vs Production Mode

**Demo Mode** (Default):
- Uses realistic synthetic data for training and demonstration
- Safe for testing and learning the system
- No real candidate or company information

**Production Mode** (When real data is available):
- Uses actual historical employment data
- Requires proper data setup by IT administrator
- Provides real matching recommendations

---

## 🖥️ Main Interface Tour

### Dashboard Layout

The system interface is organized into four main tabs:

#### 1. 🔍 Ricerca Candidato (Candidate Search)
- **Purpose**: Find company matches for individual candidates
- **Primary Use**: Daily placement operations
- **Users**: Case managers, placement coordinators

#### 2. 📊 Analytics
- **Purpose**: View system statistics and performance metrics
- **Primary Use**: Monitoring and reporting
- **Users**: Supervisors, data analysts

#### 3. 📋 Dataset
- **Purpose**: Browse and export candidate/company data
- **Primary Use**: Data verification and management
- **Users**: Data administrators, quality control

#### 4. ℹ️ Info Sistema (System Info)
- **Purpose**: System status and technical information
- **Primary Use**: Troubleshooting and configuration
- **Users**: IT support, system administrators

### Sidebar Configuration Panel

Located on the left side of the interface:

**🔧 Configurazione Sistema**:
- **Model Selection**: Choose AI model (if multiple available)
- **Attitude Threshold**: Minimum employment readiness (0.0-1.0)
- **Maximum Distance**: Search radius in kilometers (5-50 km)
- **Top Recommendations**: Number of results to show (3-10)

---

## 🔍 Finding Matches for Candidates

### Step 1: Access Candidate Search

1. Click on the **"🔍 Ricerca Candidato"** tab
2. You'll see two columns:
   - **Left**: Candidate data input
   - **Right**: Company recommendations (initially empty)

### Step 2: Input Candidate Information

You have two options for entering candidate data:

#### Option A: Use Existing Candidate

1. **Check the box** "Usa candidato esistente" (Use existing candidate)
2. **Select from dropdown**: Choose from pre-loaded candidates
3. **Review auto-filled data**: System automatically populates all fields
4. **Verify information**: Ensure data is correct and current

#### Option B: Manual Input

1. **Leave unchecked** "Usa candidato esistente"
2. **Fill in candidate details**:

   **Basic Information**:
   - **Area Residenza**: Select candidate's residential area
   - **Titolo Studio**: Choose education level
   - **Tipo Disabilità**: Select disability type

   **Assessment Scores**:
   - **Attitudine**: Employment readiness (0.0-1.0 scale)
     - 0.0-0.3: Low readiness
     - 0.4-0.6: Moderate readiness  
     - 0.7-1.0: High readiness

   **Experience Data**:
   - **Anni Esperienza**: Years of work experience
   - **Mesi Disoccupazione**: Months unemployed

   **Exclusions**:
   - **Esclusioni**: Enter work limitations (comma-separated)
   - Examples: "Turni notturni, Lavori in quota"

### Step 3: Configure Search Parameters

**Adjust settings in sidebar if needed**:

- **Soglia Attitudine**: Lower for broader search, higher for quality
- **Distanza Max**: Expand for more options, reduce for local focus
- **Top Raccomandazioni**: More results for comprehensive review

### Step 4: Execute Search

1. **Click** "🔄 Trova Aziende Compatibili" button
2. **Wait for processing**: Usually takes 2-5 seconds
3. **Review results**: System displays ranked recommendations

### Understanding Search Process

The system performs these steps automatically:

1. **Attitude Filter**: Excludes candidates below minimum threshold
2. **Geographic Filter**: Only considers companies within distance limit
3. **Compatibility Analysis**: Uses AI to match exclusions vs company activities
4. **Multi-factor Scoring**: Combines compatibility, distance, attitude, and company factors
5. **Ranking**: Orders results by final matching score

---

## 📊 Understanding Results

### Results Display Format

Each recommendation shows:

**Company Header**:
- **Company Name** and **Overall Score** (percentage)
- Visual score indicator (higher = better match)

**Key Metrics Row 1**:
- **Sector**: Type of business activity
- **Distance**: Kilometers from candidate residence
- **Employees**: Company size

**Key Metrics Row 2**:
- **Compatibility**: Semantic match score (percentage)
- **Remote**: Remote work availability
- **Positions**: Open positions for disabled candidates

### Score Interpretation

**Overall Score Ranges**:
- **85-100%**: Excellent match - highly recommended
- **70-84%**: Good match - suitable for placement
- **55-69%**: Fair match - may require additional evaluation
- **Below 55%**: Poor match - not recommended

**Compatibility Score**:
- **90-100%**: No conflicts found between exclusions and job requirements
- **70-89%**: Minor potential conflicts - interview recommended
- **50-69%**: Some conflicts present - careful evaluation needed
- **Below 50%**: Significant conflicts - likely incompatible

### Visual Analytics

**Score Distribution Chart**:
- Bar chart showing relative scores across all recommendations
- Helps identify clear winners vs close competitions

**Distance vs Compatibility Scatter Plot**:
- Shows trade-offs between proximity and job fit
- Larger circles indicate higher overall scores

### Results Actions

**No Results Found**:
If no companies appear:
1. **Increase distance** threshold in sidebar
2. **Lower attitude** threshold if appropriate
3. **Review exclusions** - may be too restrictive
4. **Check candidate location** - ensure it's valid

---

## 📊 Analytics Dashboard

### System Overview Metrics

**Key Performance Indicators**:
- **👥 Candidati Totali**: Total candidates in system
- **🏢 Aziende Totali**: Total companies available
- **📈 Attitudine Media**: Average employment readiness across candidates
- **💼 Posizioni Aperte**: Total open positions system-wide

### Distribution Charts

**Disability Types Distribution**:
- Shows breakdown of candidate disability categories
- Helps identify service focus areas
- Useful for resource planning

**Company Sectors Distribution**:
- Displays variety of available employment sectors
- Identifies placement opportunities by industry
- Guides business development efforts

### Using Analytics for Operations

**Daily Monitoring**:
- Check open positions vs candidate volume
- Monitor average attitude scores for trending
- Identify sectors with highest opportunity

**Strategic Planning**:
- Use disability distribution for specialized programs
- Target company outreach based on sector gaps
- Plan training programs based on compatibility patterns

---

## 📋 Dataset Management

### Viewing Candidate Data

1. **Navigate to** "📋 Dataset" tab
2. **Select** "Candidati" radio button
3. **Review data table**:
   - All candidate records with complete information
   - Sortable columns for data exploration
   - Search functionality for specific records

**Key Columns Explained**:
- **ID_Candidato**: Unique identifier
- **Score Attitudine al Collocamento**: Employment readiness (0.0-1.0)
- **Years_of_Experience**: Professional experience
- **Durata Disoccupazione**: Unemployment duration (months)
- **Esclusioni**: Work limitations from medical evaluation

### Viewing Company Data

1. **Select** "Aziende" radio button
2. **Review company information**:
   - Business details and contact information
   - Compatibility descriptions and requirements
   - Geographic and size information

**Key Columns Explained**:
- **Nome Azienda**: Company identifier
- **Tipo di Attività**: Business sector/activity
- **Compatibilità**: Description of suitable disability accommodations
- **Posizioni Aperte**: Available positions for disabled candidates
- **Remote**: Remote work availability (0=No, 1=Yes)
- **Certification**: Disability-friendly certification status

### Data Export Functions

**Candidate Data Export**:
1. **Click** "📥 Scarica CSV Candidati" button
2. **Save file** to desired location
3. **Use for**: External analysis, reporting, backup

**Company Data Export**:
1. **Click** "📥 Scarica CSV Aziende" button
2. **File includes**: All company information and availability
3. **Use for**: Partner outreach, capacity planning

### Data Quality Verification

**Regular Checks**:
- Verify candidate exclusions are current and accurate
- Confirm company position availability
- Update geographic information if companies relocate
- Review compatibility descriptions for accuracy

---

## ⚙️ Configuration Settings

### Threshold Adjustments

**Attitude Threshold** (Soglia Attitudine):
- **Default**: 0.3 (30%)
- **Lower (0.1-0.2)**: Include candidates with lower readiness
- **Higher (0.4-0.6)**: Focus on most employment-ready candidates
- **Impact**: Affects candidate pool size

**Distance Threshold** (Distanza Max):
- **Default**: 30 km
- **Urban areas**: 20-25 km for local focus
- **Rural areas**: 40-50 km for adequate options
- **Impact**: Balances commute feasibility vs opportunity variety

**Top Recommendations**:
- **Default**: 5 results
- **Fewer (3)**: Quick decision-making
- **More (7-10)**: Comprehensive evaluation
- **Impact**: Analysis depth vs simplicity

### Advanced Configuration

**Model Selection** (if available):
- Choose between different AI models
- Each model may have different strengths
- Default selection is usually optimal

**When to Adjust Settings**:

**Expand Search** when:
- Few or no results for qualified candidates
- Rural locations with limited local options
- Specialized disability requirements

**Narrow Search** when:
- Too many low-quality matches
- Need to focus on highest-probability placements
- Time constraints require quick decisions

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Problem: No Results Found
**Symptoms**: "No companies found" message appears
**Solutions**:
1. **Increase distance** threshold to 40-50 km
2. **Lower attitude** threshold to 0.2-0.3
3. **Review exclusions** - ensure they're not overly restrictive
4. **Check location** - verify candidate area is valid Italian location

#### Problem: All Scores Very Low
**Symptoms**: All recommendations below 60%
**Solutions**:
1. **Review exclusions accuracy** - may be too broad or incorrectly entered
2. **Check compatibility descriptions** - companies may need updated information
3. **Consider lower thresholds** - current settings may be too strict

#### Problem: System Loading Slowly
**Symptoms**: Interface takes >30 seconds to respond
**Solutions**:
1. **Refresh browser** page
2. **Clear browser cache** and reload
3. **Check internet connection** speed
4. **Try different browser** if issues persist

#### Problem: Geographic Errors
**Symptoms**: "Distance calculation failed" or unrealistic distances
**Solutions**:
1. **Verify address format** - use "City, Province, Italy" format
2. **Check spelling** of Italian city names
3. **Use major cities** instead of small villages if issues persist

### Getting Technical Support

**Before Contacting Support**:
1. **Note exact error message** if any appears
2. **Record steps** that led to the problem
3. **Check system status** in "Info Sistema" tab
4. **Try basic solutions** listed above

**Contact Information**:
- **Technical Support**: michele.melch@gmail.com
- **Academic Support**: oleksandr.kuznetsov@uniecampus.it
- **Include in email**: Screenshots, error messages, steps to reproduce

---

## 🎯 Best Practices

### Daily Operations

**Morning Routine**:
1. **Check system status** in Info Sistema tab
2. **Review analytics** for any overnight changes
3. **Verify top priority candidates** have current information

**Candidate Processing**:
1. **Always verify exclusions** with candidate before searching
2. **Use existing candidate data** when available for consistency
3. **Document successful placements** for system improvement

**Result Evaluation**:
1. **Focus on top 3 recommendations** for initial outreach
2. **Consider geographic preferences** even with high scores
3. **Review compatibility details** beyond just the score

### Weekly Reviews

**Data Quality**:
- Update candidate information based on new assessments
- Verify company position availability and requirements
- Remove or update inactive companies

**Performance Analysis**:
- Review successful vs unsuccessful placement patterns
- Identify companies with highest placement success
- Note any systematic issues with recommendations

### Integration with Existing Workflow

**CPI Integration**:
1. **Use system for initial screening** of candidates
2. **Combine with manual evaluation** for final decisions
3. **Document placement outcomes** for continuous improvement

**SIL Coordination**:
1. **Share recommendations** with case managers
2. **Coordinate follow-up** on high-score matches
3. **Track long-term placement success**

### Quality Assurance

**Recommendation Validation**:
- **Cross-check exclusions** against company requirements manually for top matches
- **Verify company information** before making contact
- **Confirm candidate preferences** align with recommendations

**Continuous Improvement**:
- **Track placement success rates** by score ranges
- **Report systematic issues** to technical team
- **Suggest improvements** based on field experience

---

## 📞 Support and Resources

### Quick Reference

**Key Shortcuts**:
- **Tab Navigation**: Use browser tabs for multiple candidates
- **Sidebar Settings**: Adjust thresholds without page reload
- **Export Functions**: Available in Dataset tab for all data

**Important Thresholds**:
- **Attitude**: 0.3 default (adjust based on candidate pool)
- **Distance**: 30 km default (expand for rural areas)
- **Compatibility**: 50% minimum for viable placement

### Training Resources

**New User Training**:
1. **Start with demo mode** to understand interface
2. **Practice with test candidates** before real operations
3. **Review this guide** section by section

**Advanced Features**:
- **Analytics interpretation** for strategic planning
- **Configuration optimization** for different scenarios
- **Integration techniques** with existing CPI/SIL workflows

### Feedback and Improvement

**How to Provide Feedback**:
- **Email suggestions** to michele.melch@gmail.com
- **Report bugs** with detailed reproduction steps
- **Share success stories** to help improve the system

**What Feedback Helps**:
- Real-world placement outcomes vs system recommendations
- Interface usability suggestions
- Additional features that would improve operations
- Integration challenges with existing systems

---

*This User Guide is designed to help employment professionals maximize the effectiveness of the Disability Job Matching System. For additional support or specific questions about your implementation, please contact the development team.*

---

**Document Version**: 1.0  
**Last Updated**: June 2025  
**Next Review**: December 2025