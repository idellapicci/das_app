# dash-app
Testing creating dash app for a simple descriptive analysis

## Running

Cd into your dash-app folder.

Build the image running the command:

```docker build -f Dockerfile -t dash_app .```

Running a container:

```docker run -d -it -p 80:8050 dash_app:latest ```

Finally open your brower on localhost to access the app.

## Sources

### Data Sample

You can upload the sensor sample csv from data folder for testing. 
Obs: free data obtained from https://www.kaggle.com/nphantawee/pump-sensor-data.

### Dash App

Dash ploty documentation: https://dash.plotly.com/

Dash App Gallery with visual and code examples you can check to learn more: https://dash-gallery.plotly.host/Portal/




