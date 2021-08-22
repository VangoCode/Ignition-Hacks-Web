from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import check_cat


UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['jpg'])

app = Flask(__name__)

# Set upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def start():
    return render_template("index.html")

@app.route("/result", methods=['GET', 'POST'])
def result():
    if request.method == "POST":
        # check if the post request has the file part
        if 'cat_upload' not in request.files:
            return render_template("/apology.html")
        file = request.files['cat_upload']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return render_template("/apology.html")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            breed = check_cat.callImage(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template("result.html", file=file, filename=filename, breed=breed)
    return render_template("apology.html")


#app.run()