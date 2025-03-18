 


from django import forms

class ImageSearchForm(forms.Form):
    image_folder = forms.CharField(label="Image Folder Path", max_length=255)
    image = forms.ImageField(label="Upload Query Image")

class GroupImageSearchForm(forms.Form):
    image_folder = forms.CharField(label="Image Folder Path", max_length=255)
    image = forms.ImageField(label="Upload Query Image")