def auth_middleware(request, call_next):
    """
    Authentication middleware for API requests.
    
    Args:
        request: The incoming request
        call_next: The next middleware function to call
    
    Returns:
        The response after processing the request
    """
    # Extract API key from header
    api_key = request.headers.get("Authorization")
    
    # Validate API key
    if api_key and api_key.startswith("Bearer "):
        # Extract the token
        token = api_key.replace("Bearer ", "")
        
        # TODO: Implement actual token validation
        if token != "":
            request.state.user = {"id": "user_1", "role": "admin"}
        else:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key"}
            )
    else:
        # Allow public endpoints or return 401
        if request.url.path.startswith("/public"):
            pass
        else:
            return JSONResponse(
                status_code=401,
                content={"detail": "API key required"}
            )
    
    # Continue processing the request
    return call_next(request)
