import requests

from agent_marketplace.config import get_settings

def get_coordinates_from_address(address):
    """
    Function to get coordinates from an address using Google Geocoding API
    
    Args:
        address (str): The address to geocode
        
    Returns:
        dict: Dictionary containing latitude, longitude, and formatted address
        
    Raises:
        Exception: If the geocoding request fails
    """

    settings = get_settings()
    
    try:
        # Construct the API URL
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        
        # Parameters for the request
        params = {
            "address": address,
            "key": settings.google_api_key
        }
        
        # Make the API request
        response = requests.get(url, params=params)
        data = response.json()
        
        # Check if the request was successful
        if data["status"] != "OK":
            error_message = data.get("error_message", "")
            raise Exception(f"Geocoding error: {data['status']}. {error_message}")
        
        # Extract coordinates from the response
        location = data["results"][0]["geometry"]["location"]
        
        return {
            "lat": location["lat"],
            "lng": location["lng"],
            "formatted_address": data["results"][0]["formatted_address"]
        }
        
    except Exception as e:
        print(f"Error getting coordinates: {e}")
        raise

if __name__ == "__main__":
    print(get_coordinates_from_address("One Apple Park Way, Cupertino, CA"))
