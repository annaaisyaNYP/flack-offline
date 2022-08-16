from wtforms import Form, StringField, FileField, validators

class Input(Form):
    text = StringField('',[validators.DataRequired()])