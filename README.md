# Glance 
[![HitCount](http://hits.dwyl.com/{username}/{project-name}.svg)](http://hits.dwyl.com/{username}/{project-name})
[![Collaborators](https://img.shields.io/badge/collaborators-0-orange.svg)](https://github.com/gianfa/Glance/issues)
  

A slim Pandas extension for having a quick Glance at datasets.

## Why
When you get a new dataset there are many first operations you do and, often, they are quite the same. For this reason I thought about writing this little module along the way, expanding such wonderful Pandas package.
It allows to easily perform some starting operations like to scan your dataset fields looking for typical types.
<br><br>
It scans fields to indentify:
- [x] Emails
- [x] Datetimes
- [ ] Urls to an image

## How to install it
As you can see, at the moment, is a simple module that you can move into your working folder and import.

## How to use it
Here is an example<br>
        <br>>>>df = pd.read_csv( filename )
        <br>>>>df = df.glance.glance()
        <br>>>>df.glance.notnull()

## Give feedbacks!
Feel free to contribute or report issues/errors!
