import os
import platform

def setup_proxy_env():
    """Set proxy environment variables for Windows and Linux."""
    proxy_vars = {
        "http_proxy": "http://proxy-dmz.intel.com:912",
        "https_proxy": "http://proxy-dmz.intel.com:912",
        "ftp_proxy": "http://proxy-dmz.intel.com:912",
        "socks_proxy": "http://proxy-dmz.intel.com:1080",
        "no_proxy": "intel.com,.intel.com,localhost,127.0.0.1,10.0.0.0/8,192.168.0.0/16,172.16.0.0/12",
    }
    # Set lowercase and uppercase variants
    for key, value in proxy_vars.items():
        os.environ[key] = value
        os.environ[key.upper()] = value

# Call this function at the start of your main script to enforce proxy settings
