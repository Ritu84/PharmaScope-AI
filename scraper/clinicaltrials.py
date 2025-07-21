import requests
import logging
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_clinicaltrials_data(target: str):
    base_url = "https://clinicaltrials.gov/api/int/studies"
    params = {
        "cond": target,
        "agg.synonyms": "true",
        "aggFilters": "phase:0 1 2 3 4",
        "checkSpell": "true",
        "from": 0,
        "limit": 100,
        "fields": "OverallStatus,LastKnownStatus,StatusVerifiedDate,HasResults,BriefTitle,Condition,InterventionType,InterventionName,LocationFacility,LocationCity,LocationState,LocationCountry,LocationStatus,LocationZip,LocationGeoPoint,LocationContactName,LocationContactRole,LocationContactPhone,LocationContactPhoneExt,LocationContactEMail,CentralContactName,CentralContactRole,CentralContactPhone,CentralContactPhoneExt,CentralContactEMail,Gender,MinimumAge,MaximumAge,StdAge,NCTId,StudyType,LeadSponsorName,Acronym,EnrollmentCount,StartDate,PrimaryCompletionDate,CompletionDate,StudyFirstPostDate,ResultsFirstPostDate,LastUpdatePostDate,OrgStudyId,SecondaryId,Phase,LargeDocLabel,LargeDocFilename,PrimaryOutcomeMeasure,SecondaryOutcomeMeasure,DesignAllocation,DesignInterventionModel,DesignMasking,DesignWhoMasked,DesignPrimaryPurpose,DesignObservationalModel,DesignTimePerspective,LeadSponsorClass,CollaboratorClass",
        "columns": "conditions,interventions,collaborators",
        "highlight": "true",
        "sort": "@relevance"
    }
   
    try:
        response = requests.get(base_url, params=params, headers={"Accept": "application/json"})
        response.raise_for_status()
        data = response.json()
        
        output_dir = "output/scraper-results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "clinical-trails.json")
        with open(output_path, "w") as f:
            f.write(json.dumps(data, indent=2))
        
        return data

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return None


# def main():
#     target = "CD47"
#     studies = fetch_clinicaltrials_data(target)
#     print(f"\nFetched {len(studies)} studies for target '{target}'")
#     print(studies)
#     with open("clinicaltrails.json", "w") as f:
#         f.write(json.dumps(studies, indent=2))

# if __name__ == "__main__":
#     main()