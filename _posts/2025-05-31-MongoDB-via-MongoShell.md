---
layout: post
title:  "MongoDB via Mongo Shell"
date:   2025-05-31 00:00:00 +0000
categories: MongoDB Bash Python
---


<img src="https://github.com/pw598/pw598.github.io/blob/main/_posts/images/mg1.jpg?raw=true" style="height: 350px; width:auto;">


This is the first of 3 articles on MongoDB and the power of unstructured databases. The focus is on the Mongo shell, though parallel resources linked to within utilize command-line (Bash) and Python (PyMongo) commands.



# Outline

1. Introduction
2. Installation
3. The Mongo Shell
4. Getting Started
5. Show Databases
6. Import Data
7. Select Imported Database
8. Show Collections
9. Sample Documents
10. Get Record Counts
11. Get List of Distinct Fields
12. Get Number of Distinct Values Per Field
13. CRUD Operations
    - Remove and Create
    - Read
    - Update
14. Indexes
    - View Indexes
    - Create Indexes
15. What's Next?




# Introduction

Unstructured databases have greatly increased in popularity over the past 15 years, addressing the need to manage increasingly large, diverse, and evolving datasets. The lack of adherence to a rigid schema allows data to be stored in variable formats, rather than pre-specified columns, and this flexibility is well-suited toward modern data sources such as multimedia with metadata, text, and embedded or hierarchical data. Normalization and joins are avoided, lending toward the ability to horizontally scale compute resources, and this is heavily relied upon by organizations who deal with massive amounts of web transactions.

MongoDB is the most popular of the unstructured databases, with a large developer community, integration with a multitude of APIs, and a cloud service called MongoDB Atlas. 

<img src="https://raw.githubusercontent.com/pw398/pw398.github.io/refs/heads/main/_posts/images/1-1.png" style="height: 250px; width:auto;">

The native language for command-line instructions is Javascript, however the Mongo shell provides its own simplified language. The 'documents', a term analogous to records in a structured database, are in a JSON-like format, and are organized into 'collections', the analog to a table.

In this article, we will focus on making commands through the Mongo shell, which is the simplest method. However, parallel notebooks utilizing the command line (Bash) and Python (PyMongo) are linked to below.
- <a href="https://github.com/pw598/Articles/blob/main/notebooks/MongoDB-via-Mongo-Shell.ipynb">Mongo Shell Notebook</a>
- <a href="">Bash Shell Notebook - Coming Soon</a>
- <a href="">PyMongo Workbook - Coming Soon</a>

Subsequent articles will utilize PyMongo. The content of this article will provide an overview of querying and database operations, the second will focus on aggregation pipelines, and the third will focus on deploying machine learning upon streaming text data.



# Installation

The MongoDB website has robust tutorials for installation. Be sure to get the Mongo shell and add it to PATH so you can follow along with the below. Although the code is provided in a notebook format, most commands in this article and the first 'notebook' will only work through the Mongo shell. This can be opened directly (by clicking on the .exe file), or from the command prompt using <code>mongosh</code>.

- <a href="https://www.mongodb.com/docs/manual/installation/">MongoDB Installation Tutorials</a>
- <a href="https://www.mongodb.com/try/download/shell">MongoDB Shell</a>

Also, regardless of which language or platform you plan on using, be sure to get the MongoDB command line tools, as this will be essential toward actions like reading and writing to file.

- <a href="https://www.mongodb.com/try/download/database-tools">MongoDB Command Line Tools</a>

For PyMongo, if you are using Anaconda, I recommend using <code>conda install pymongo</code> from the Anaconda command prompt (activate it with <code>conda activate base</code> first if operating from the general command prompt). Otherwise, review the instructions <a href="https://www.mongodb.com/docs/languages/python/pymongo-driver/upcoming/get-started/download-and-install/?msockid=10b2dcbcd0206ffd212ec970d1946ee7">here</a>.



# The Mongo Shell

The capabilities of the Mongo shell include performing CRUD (create, read, update, delete) operations, querying and index management, user and database administration, and Javascript support. The commands are simpler and less verbose than calling upon MongoDB through the APIs, however the Mongo shell is not optimized for large-scale data processing, so APIs like PyMongo will perform better in this regard.

Multi-line commands can be entered by pressing Enter to move to the next line, and then ctrl+Enter when ready to execute. This can be cumbersome for complex commands, but you can use <code>load()</code> to execute code from a Javascript file, or paste in multiple lines at once.



# Getting Started

### Opening the MongoDB Shell (mongosh)

### From the Command Line

Once installed, we can open the Mongo shell simply by typing <code>mongosh</code> (or the appropriate environment variable name) into the command prompt. 

```bash
mongosh
```
This will default to the localhost server. We can specify an alternative upon opening by using the command:

```bash
mongosh --host <hostname> --port <port>
```


### Opening Directly

If opening the Mongo shell by clicking on the .exe. file, it will prompt for a server, suggesting <code>localhost</code> by default.

```js
// Please enter a MongoDB connection string (Default: mongodb://localhost/):</code>
```

I will go with <code>localhost</code>. We can type in the string it suggested, or simply press Enter.

```bash
mongodb://localhost/
```


# Show Databases

We can use <code>show dbs</code> or <code>show databases</code> to get the list of existing databases. The three listed below are system-related, and come with the installation.

```js
show dbs
```

<p></p>

```js
// admin        132.00 KiB
// config       116.00 KiB
// local         96.00 KiB
```



# Import Data 

We will be importing clickstream data from a <code>.bson</code> file with the data records, along with a <code>.json</code> file with a single record of metadata. It corresponds to the web traffic of an e-commerce store called Kirana Store, and indicates with the <code>Activity</code> field whether a pageload or click on a product occured.

Our import commands below will specify <code>--drop</code> to drop the database first if it currently exists, but if at any point you wish to drop a database (in this example named <code>clickstream</code>), you can use the command <code>use clickstream</code> to select it, followed by <code>db.dropDatabase()</code>.


### Drop <code>clickstream</code> if Exists (Optional)

```js
use clickstream
```

<p></p>

```js
// switched to db clickstream
```

<p></p>

```js
db.dropDatabase()
```

<p></p>

```js
// { ok: 1, dropped: 'clickstream' }
```

Although the code in this workbook is almost entirely focused on making commands through the Mongo shell, the Mongo tools such as <code>mongorestore</code> and <code>mongoimport</code> cannot be used through the Mongo shell, and must be called upon from the command line.

The syntax for these commands is evident from the last two lines below. You are free to use hard-coding and ignore all of the variable-setting done by the preceding lines. The below creates an <code>import_data.bat</code> file which can be run from the command line to import our <code>.bson</code> file using <code>mongorestore</code>, and <code>.json</code> metadata using <code>mongoimport</code>. The text following <code>REM</code> are comments.

```bash
REM the below are optional, but used to avoid hard-coding the import commands further below

@echo off
set HOST=localhost
set PORT=27017
set DBNAME=clickstream
set IMPORT_FILE_FOLDER=C:\Users\patwh\Downloads
set BSON_FILE_NAME=clicks
set JSON_FILE_NAME=clicks.metadata

set bson_file=%IMPORT_FILE_FOLDER%\%BSON_FILE_NAME%.bson
set json_file=%IMPORT_FILE_FOLDER%\%JSON_FILE_NAME%.json

set collection_bson=%BSON_FILE_NAME%
set collection_json=%JSON_FILE_NAME%

REM import operation
mongorestore --host %HOST%:%PORT% --db %DBNAME% --collection %collection_bson% --drop "%bson_file%"
mongoimport --host %HOST%:%PORT% --db %DBNAME% --collection %collection_json% --drop --type json "%json_file%"

echo Done.
```

With the <code>.bat</code> file created, simply call upon it from the command line, replacing my directory below with your own. Use the exclamation mark if in a Jupyter/Colab notebook, otherwise drop it.

```bash
!"C:/Users/patwh/Downloads/import_data.bat"
```

<p></p>

```bash 
# 2025-05-28T18:38:13.827-0600  finished restoring clickstream.clicks (6100000 documents, 0 failures)
# 2025-05-28T18:38:13.827-0600  no indexes to restore for collection clickstream.clicks
# 2025-05-28T18:38:13.827-0600  6100000 document(s) restored successfully. 0 document(s) failed to restore.
# 2025-05-28T18:38:14.488-0600  connected to: mongodb://localhost:27017/
# 2025-05-28T18:38:14.490-0600  dropping: clickstream.clicks.metadata
# 2025-05-28T18:38:14.507-0600  1 document(s) imported successfully. 0 document(s) failed to import.
```

We see that the data file contains 6.1M records, and the metadata file contains only one record. The first step toward viewing the details is to select a database. First, we use <code>show dbs</code> to confirm that our imported database exists.


# Select Imported Database

```js
show dbs
```

<p></p>

```js
// admin        132.00 KiB
// clickstream  428.30 MiB
// config       108.00 KiB
// local         96.00 KiB
```

To select it, we simply use:

```js
use clickstream
```

<p></p>

```js
// switched to db clickstream
```


# Show Collections

As mentioned above, a MongoDB database contains 'collections' of documents, i.e., records, each of which has a set of fields that may or may not exist for other documents in particular. We list the collections belonging to <code>clickstream</code> as follows.

```js
show collections
```

<p></p>

```js
// clicks
// clicks.metadata
```



# Sample Documents

To view the first document found in a collection, we use <code>db.collection.findOne()</code>. As you will see shortly, we can pass arguments into this function in order to filter.

```js
db.clicks.findOne()
```

<p></p>

```js
// {
//   _id: ObjectId('60df1029ad74d9467c91a932'),
//   webClientID: 'WI100000244987',
//   VisitDateTime: ISODate('2018-05-25T04:51:14.179Z'),
//   ProductID: 'Pr100037',
//   Activity: 'click',
//   device: { Browser: 'Firefox', OS: 'Windows' },
//   user: { City: 'Colombo', Country: 'Sri Lanka' }
// }
```

<p>The <code>_id</code> field is a unique identifier attached to each record. Duplicate IDs are not permitted, nor is deleting this field.</p>

We'll take a look at the lone record in the <code>clicks.metadata</code> collection.

```js
db.clicks.metadata.findOne()
```

<p></p>

```js
// {
//   _id: ObjectId('6837ada071d28360c34516c3'),
//   indexes: [ { v: 2, key: { _id: 1 }, name: '_id_' } ],
//   uuid: 'ee6da5fe5bdf42b2bc3cecee40723af6',
//   collectionName: 'clicks'
// }
````

We see that this contains some index information. Indexing will be covered in more detail at the end of this article. 



# Get Record Counts

If we weren't familiar with the number of records in this database upon import, we could run the following to get the count of documents.

```js
db.clicks.countDocuments()
````

<p></p>

```js
// 6100000
```

<p></p>

```js
db.clicks.metadata.countDocuments()
```

<p></p>

```js
// 1
```


# Get List of Distinct Fields

The below command is a little more involved. To get the list of distinct fields in the collection, we want to loop through each of the documents. Javascript is the native language of the Mongo shell, so if we want to create variables and loops, we must either enter Javascript commands, or use the <code>load()</code> function upon a <code>.js</code> file (or use an API like PyMongo instead of the Mongo shell). As mentioned above, the shell is not optimized for large-scale processing, so PyMongo would actually be faster at executing the below.


```js
(function() {
  var fields = {};
  db.clicks.find().limit(1000000).forEach(function(doc) {
    Object.keys(doc).forEach(function(key) {
      fields[key] = true;
    });
  });
  printjson(Object.keys(fields));
})();
```

<p></p>

```js
// [
//   '_id',
//   'webClientID',
//   'VisitDateTime',
//   'ProductID',
//   'Activity',
//   'device',
//   'user'
// ]
````

That provided a list of top-level fields, but not nested fields that reside in the hierarchy. The following script will loop through second-level fields as well. It is quite lengthy to be entered line by line, so we'll use a <code>.js</code> file, along with the <code>load()</code> command.

I said this article would be all about Mongo shell commands, but for my own sake, I will use the below Python function to generate the <code>.js</code> files and ensure proper format. This is purely optional, and only for the sake of convenience.


```js
// Function for Creating .js Files From Text in Python
import os

def save_js_commands(js_input, js_folder, js_filename):
    filepath = f"{js_folder}/{js_filename}.js"
    
    try:
        # Ensure input is a string and strip leading/trailing whitespace
        if isinstance(js_input, str):
            lines = [line.rstrip() for line in js_input.splitlines() if line.strip()]
        elif isinstance(js_input, list):
            lines = [line.rstrip() for line in js_input if line.strip()]
        else:
            raise TypeError("Input must be a string or list of strings.")

        # Write to file with explicit UTF-8 encoding and Windows line endings
        with open(filepath, 'w', encoding='utf-8', newline='\r\n') as file:
            for line in lines:
                file.write(line + '\n')

        print(f"✅ JavaScript code saved successfully to: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return None
```


The below specifies the Javascript as text, and calls upon the above function to create the <code>.js</code> file.


```python
# unique_fields_nested.js

js_code = """
(function() {
  var fields = {};

  function extractFields(obj, prefix) {
    for (var key in obj) {
      if (!obj.hasOwnProperty(key)) continue;

      var fullKey = prefix ? prefix + "." + key : key;
      fields[fullKey] = true;

      // Recurse into nested objects (not arrays or null)
      if (typeof obj[key] === "object" && obj[key] !== null && !Array.isArray(obj[key])) {
        extractFields(obj[key], fullKey);
      }
    }
  }

  db.clicks.find().limit(1000000).forEach(function(doc) {
    extractFields(doc, "");
  });

  printjson(Object.keys(fields));
})();

"""

js_folder = r"C:/Users/patwh/Downloads/js_commands"
js_filename = "unique_fields_nested"

js_file = save_js_commands(js_code, js_folder, js_filename)
exe_file = f"{js_folder}\\{js_filename}.js"
```

<p></p>

```python
# ✅ JavaScript code saved successfully to: C:/Users/patwh/Downloads/js_commands/unique_fields_nested.js
```

The below is what we type into the Mongo shell to execute the <code>.js</code> script. Replace my directory with the directory pertaining to yourself.

```js
load("C:/Users/patwh/Downloads/js_commands/unique_fields_nested.js")
````

<p></p>

```js
// [
//   '_id',
//   'webClientID',
//   'VisitDateTime',
//   'ProductID',
//   'Activity',
//   'device',
//   'device.Browser',
//   'device.OS',
//   'user',
//   'user.City',
//   'user.Country',
//   'user.UserID'
// ]
```



# Get Number of Distinct Values by Field

It would be informative to see how many distinct values correspond to each of the fields in the collection. For that, we can use the following.

```python
js_code = """
(function() {
  const collection = db.clicks;

  const fields = [
    '_id',
    'webClientID',
    'VisitDateTime',
    'ProductID',
    'Activity',
    'device',
    'device.Browser',
    'device.OS',
    'user',
    'user.City',
    'user.Country',
    'user.UserID'
  ];

  fields.forEach(field => {
    const pipeline = [
      { $group: { _id: `$${field}` } },
      { $group: { _id: null, count: { $sum: 1 } } }
    ];

    const result = collection.aggregate(pipeline).toArray();
    const count = result.length > 0 ? result[0].count : 0;
    print(`${field}: ${count} unique values`);
  });
})();

"""

js_folder = r"C:/Users/patwh/Downloads/js_commands"
js_filename = "unique_value_counts_hardcoded_fields"

js_file = save_js_commands(js_code, js_folder, js_filename)
exe_file = f"{js_folder}\\{js_filename}.js"
```

<p></p>

```python
# ✅ JavaScript code saved successfully to: C:/Users/patwh/Downloads/js_commands/unique_value_counts_hardcoded_fields.js
```

<p></p>

```js
load("C:/Users/patwh/Downloads/js_commands/unique_value_counts_hardcoded_fields.js")
```

<p></p>

```js
// _id: 6100000 unique values
// VisitDateTime: 6089023 unique values
// webClientID: 1091455 unique values
// user: 72162 unique values
// user.UserID: 34051 unique values
// user.City: 26260 unique values
// ProductID: 10938 unique values
// user.Country: 222 unique values
// device: 151 unique values
// device.Browser: 82 unique values
// device.OS: 18 unique values
// Activity: 2 unique values
```

We see some fields we didn't see in the sample document, such as <code>user.UserID</code>. These correspond to users of the Kirana store website who have signed up to create an account. 



### Dynamic Version (First Finds Fields, then Distinct Value Counts)


Finding the unique list of fields and then hard-coding them into the search for distinct values broke a very long task (given the 6.1M documents) into two shorter tasks. But the code to perform both actions in dynamic fashion, without hard-coding, is provided below.

```python
js_code = """
// Use the correct database
db = db.getSiblingDB("clickstream");

print("=== Starting Unique Field Count ===");

const SAMPLE_SIZE = 1000; // Sample more than 1 doc to detect more fields
const MAX_FIELDS = 100;   // Safety cap
let fieldSet = {};

function extractFields(obj, prefix = "") {
    for (let key in obj) {
        if (!obj.hasOwnProperty(key)) continue;
        const fullKey = prefix ? prefix + "." + key : key;
        fieldSet[fullKey] = true;
        if (typeof obj[key] === "object" && obj[key] !== null && !Array.isArray(obj[key])) {
            extractFields(obj[key], fullKey);
        }
    }
}

// Sample documents to build a broader field list
db.clicks.find().limit(SAMPLE_SIZE).forEach(doc => {
    extractFields(doc);
});

const fields = Object.keys(fieldSet).slice(0, MAX_FIELDS);
let results = [];

fields.forEach(field => {
    try {
        const pipeline = [
            { $group: { _id: `$${field}` } },
            { $group: { _id: null, count: { $sum: 1 } } }
        ];

        const res = db.clicks.aggregate(pipeline).toArray();
        const count = res.length ? res[0].count : 0;
        results.push({ field, count });
    } catch (e) {
        print(`Error counting values for ${field}: ${e.message}`);
    }
});

// Sort by count descending
results.sort((a, b) => b.count - a.count);

// Print nicely
results.forEach((r, i) => {
    print(`${i + 1}. ${r.field}: ${r.count} unique values`);
});

print("=== Done ===");
"""

js_folder = r"C:/Users/patwh/Downloads/js_commands"
js_filename = "count_unique_values_dynamic"

js_file = save_js_commands(js_code, js_folder, js_filename)
exe_file = f"{js_folder}\\{js_filename}.js"
```

<p></p>

```python
# ✅ JavaScript code saved successfully to: C:/Users/patwh/Downloads/js_commands/count_unique_values_dynamic.js
```

<p></p>

```js
load("C:/Users/patwh/Downloads/js_commands/count_unique_values_dynamic.js")
````

<p></p>

```js
// === Starting Unique Field Count ===
// 1. _id: 6100000 unique values
// 2. VisitDateTime: 6089023 unique values
// 3. webClientID: 1091455 unique values
// 4. user: 72162 unique values
// 5. user.UserID: 34051 unique values
// 6. user.City: 26260 unique values
// 7. ProductID: 10938 unique values
// 8. user.Country: 222 unique values
// 9. device: 151 unique values
// 10. device.Browser: 82 unique values
// 11. device.OS: 18 unique values
// 12. Activity: 2 unique values
// === Done ===
// true
````



# Get List of Unique Values for a Field

To get the list of unique values for a field, we use <code>db.collection.distinct("fieldname")</code>, which quickly returns the results. Below, we will get the list of unique browsers. The list is long, so the results below are truncated.


### Get Unique Values for <code>device.Browser</code>

```js
db.clicks.distinct("device.Browser")
```

<p></p>

```js
// [
//   'AdsBot-Google',
//   'AhrefsBot',
//   'Amazon Silk',
//   'Android',
//   'AppEngine-Google',
//   'Apple Mail',
//   'BingPreview',
//   'BlackBerry WebKit',
//   ...
//]
```



# CRUD Operations

Fundamental database operations include creating new records, removing records, updating records, and deleting records - hence the acronym CRUD. The below will demonstrate examples of each.


## Remove and Create

To demonstrate the actions of saving a record to a Javascript variable, removing a record, and creating a record, we will capture the last record's data in a variable, remove that record, and then re-insert it from the variable.


### Remove and Re-Insert the Last Record

To capture the data of the <code>clicks</code> record that we will momentarily delete, we use the <code>find()</code> operation combined with <code>sort</code> and <code>limit</code>.

```js
// capture data in a javascript variable
var lastDoc = db.clicks.find().sort({ _id: -1 }).limit(1).next();
````

The data is represented in JSON format.

```js
// {
//   _id: ObjectId('60df129dad74d9467ceebd51'),
//   webClientID: 'WI100000118333',
//   VisitDateTime: ISODate('2018-05-26T11:51:44.263Z'),
//   ProductID: 'Pr101251',
//   Activity: 'click',
//   device: { Browser: 'Chrome', OS: 'Windows' },
//   user: { City: 'Vijayawada', Country: 'India' }
// }
```

Next, we use <code>deleteOne()</code> to remove it from the collection.

```js
// remove the record from the collection
db.clicks.deleteOne({ _id: lastDoc._id });
```

<p></p>

```js
// { acknowledged: true, deletedCount: 1 }
````

Finally, we use <code>insertOne()</code> with reference to our stored variable in order to re-insert the record. If the data were not in JSON format, we would need to transform it first.

```js
// insert the record back into the collection
db.clicks.insertOne(lastDoc);
```
<p></p>

```js
// {
//   acknowledged: true,
//   insertedId: ObjectId('60df129dad74d9467ceebd51')
// }
````


### Remove and Re-Insert the Last 5 Records

Similarly, we can do the same as above with a batch of records. I will use the last 5.

We capture the data in a variable:

```js
// capture data in a javascript variable
var lastDocs = db.clicks.find().sort({ _id: -1 }).limit(5).toArray();
var idsToDelete = lastDocs.map(doc => doc._id);
```

Then, apply a delete operation to remove them from the collection:

```js
// remove the records from the collection
db.clicks.deleteMany({ _id: { $in: idsToDelete } });
```

<p></p>

```js
// { acknowledged: true, deletedCount: 5 }
```

And finally, use <code>insertMany()</code> to insert them all back in at the same time.

```js
// insert them back in
db.clicks.insertMany(lastDocs);
````

<p></p>

```js
// {
//   acknowledged: true,
//   insertedIds: {
//     '0': ObjectId('60df129dad74d9467ceebd51'),
//     '1': ObjectId('60df129dad74d9467ceebd50'),
//     '2': ObjectId('60df129dad74d9467ceebd4f'),
//     '3': ObjectId('60df129dad74d9467ceebd4e'),
//     '4': ObjectId('60df129dad74d9467ceebd4d')
//   }
// }
```



## Read

<p>We've already executed a fair amount of queries, but the below will elaborate. First, we'll filter to a particular field, in this case the <code>_id</code>. No documents have duplicate IDs, so the below <code>findOne()</code> will return either one document or none.</p>


<h3>Filter to <code>_id</code> Equal to <code>60df129dad74d9467ceebd51</code></h3>

```js
db.clicks.findOne({ _id: ObjectId("60df129dad74d9467ceebd51") });
```

<p></p>

```js
// {
//   _id: ObjectId('60df129dad74d9467ceebd51'),
//   webClientID: 'WI100000118333',
//   VisitDateTime: ISODate('2018-05-26T11:51:44.263Z'),
//   ProductID: 'Pr101251',
//   Activity: 'click',
//   device: { Browser: 'Chrome', OS: 'Windows' },
//   user: { City: 'Vijayawada', Country: 'India' }
// }
````



### Find First Record Where <code>device.Browser</code> is not Firefox

<p>If multiple records meet the specified criteria of a <code>findOne()</code> query, the first record encountered will be returned. Below, we simply replace the above criteria of having a particular <code>_id</code> with the criteria of having Firefox browser.</p>


```js
db.clicks.findOne({ "device.Browser": "Firefox" });
```

<p></p>

```js
// {
//   _id: ObjectId('60df1029ad74d9467c91a932'),
//   webClientID: 'WI100000244987',
//   VisitDateTime: ISODate('2018-05-25T04:51:14.179Z'),
//   ProductID: 'Pr100037',
//   Activity: 'click',
//   device: { Browser: 'Firefox', OS: 'Windows' },
//   user: { City: 'Colombo', Country: 'Sri Lanka' }
// }
````


### Find First 2 Records Where <code>device.Browser</code> is not Firefox

If wanting to return more than one document, we use <code>find()</code> rather than <code>findOne()</code>. As we did above, we use <code>limit</code> to truncate the data returned to a certain number of records - in this case, the first two records where the browser is not equal to <code>Firefox</code>, using the <code>$ne</code> (not equal) operator.

```js
db.clicks.find({ "device.Browser": { $ne: "Firefox" } }).limit(2);
```

<p></p>

```js
// [
//   {
//     _id: ObjectId('60df1029ad74d9467c91a933'),
//     webClientID: 'WI10000061461',
//     VisitDateTime: ISODate('2018-05-25T05:06:03.700Z'),
//     ProductID: 'Pr100872',
//     Activity: 'pageload',
//     device: { Browser: 'Chrome Mobile', OS: 'Android' },
//     user: {}
//   },
//   {
//     _id: ObjectId('60df1029ad74d9467c91a934'),
//     webClientID: 'WI10000075748',
//     VisitDateTime: ISODate('2018-05-17T11:51:09.265Z'),
//     ProductID: 'Pr100457',
//     Activity: 'click',
//     device: { Browser: 'Chrome', OS: 'Linux' },
//     user: { City: 'Ottawa', Country: 'Canada' }
//   }
// ]
```



### Find First 2 Records Where <code>VisitDateTime</code> is Greater Than 5/20/2018


Of course, we also have comparison operators such as <code>$gt</code>, used below to get two records (using <code>limit</code>) which have a date later than May 20.


```js
db.clicks.find({
  VisitDateTime: { $gt: new Date("2018-05-20") }
}).limit(2);
````

<p></p>

```js
// [
//   {
//     _id: ObjectId('60df1029ad74d9467c91a932'),
//     webClientID: 'WI100000244987',
//     VisitDateTime: ISODate('2018-05-25T04:51:14.179Z'),
//     ProductID: 'Pr100037',
//     Activity: 'click',
//     device: { Browser: 'Firefox', OS: 'Windows' },
//     user: { City: 'Colombo', Country: 'Sri Lanka' }
//   },
//   {
//     _id: ObjectId('60df1029ad74d9467c91a933'),
//     webClientID: 'WI10000061461',
//     VisitDateTime: ISODate('2018-05-25T05:06:03.700Z'),
//     ProductID: 'Pr100872',
//     Activity: 'pageload',
//     device: { Browser: 'Chrome Mobile', OS: 'Android' },
//     user: {}
//   }
// ]
```



### Get the Minimum and Maximum <code>VisitDateTime</code>


To get the minimum and maximum values of a field that spans a numerical or date-based range, we can use something like the following. Aggregation will be covered further in the following articles.


```js
db.clicks.aggregate([
  { $group: {
      _id: null,
      minVisitDateTime: { $min: "$VisitDateTime" },
      maxVisitDateTime: { $max: "$VisitDateTime" }
    } }
]);
````

<p></p>

```js
// [
//   {
//     _id: null,
//     minVisitDateTime: ISODate('2018-05-07T00:00:01.190Z'),
//     maxVisitDateTime: ISODate('2018-05-27T23:59:59.576Z')
//   }
// ]
````



### Get Count of Records Where <code>VisitDateTime</code> is Greater Than 5/20/2018


Above, we used <code>countDocuments()</code> to get the count of records in a collection. Below, we apply a filter such as the one we used just above, and find that about 2.45M records have a date greater than May 20.


```js
db.clicks.countDocuments({
  VisitDateTime: { $gt: new Date("2018-05-20") }
});
````

<p></p>

```js
// 2453050
````



### Get Count of Records Where <code>user.Country</code> is <code>India</code> or <code>United States</code>

Below, we'll focus on logical and array operators, many of which have identical counterparts in SQL. We'll see that there is often an array-operator equivalent to a logical-operator, though array operators can be favorable toward minimizing typing, particularly when there is a long list of items.

First, we'll use the <code>$or</code> operator to get the count of records when filtered to where the country is either India or United States.


#### Using <code>$or</code>

```js
db.clicks.countDocuments({
  $or: [
    { "user.Country": "India" },
    { "user.Country": "United States" }
  ]
});
````

<p></p>

```js
// 3497232
````



#### Using <code>$in</code>

Below, we get the same result by using the <code>$in</code> operator, and passing in the list of countries for which we will consider the corresponding records.


```js
db.clicks.countDocuments({
  "user.Country": { $in: ["India", "United States"] }
});
````

<p></p>

```js
// 3497232
```



### Get Count of Records Where <code>user.Country</code> is Neither <code>India</code> Nor <code>United States</code>

Now, let's do the opposite. How many users are in neither India or the United States?



#### Using <code>$and</code>

```js
db.clicks.countDocuments({
  $and: [
    { "user.Country": { $ne: "India" } },
    { "user.Country": { $ne: "United States" } }
  ]
});
```

<p></p>

```js
// 2602768
```



#### Using `$not` and `$in`

We can combine <code>$not</code> with <code>$in</code> to get the count of users that are 'not in' the list of countries provided.


```js
db.clicks.countDocuments({
  "user.Country": { $not: { $in: ["India", "United States"] } }
});
```

<p></p>

```js
// 2602768
```


#### Using <code>$nin</code>

Unlike SQL, we also have a <code>$nin</code> shortcut which stands for 'not in'.


```js
db.clicks.countDocuments({
  "user.Country": { $nin: ["India", "United States"] }
});
```

<p></p>

```js
// 2602768
```



### Get Count of Records with <code>user.UserID</code>


The e-commerce store is likely very interested in the amount of traffic which corresponds to people who have created an account. Below, we count the instances of <code>user.UserID</code>, and find that only about a tenth of traffic corresponds to this.


```js
db.clicks.countDocuments({
  "user.UserID": { $exists: true, $ne: null }
});
````

<p></p>

```js
// 602293
```



## Update

Now for some document-update operations.



### Update <code>device.Browser</code> for Record <code>60df129dad74d9467ceebd51</code> to <code>Firefox</code>

To update a record, using a toy example, we'll use the <code>$set</code> operator along with <code>updateOne</code> to set the browser of the record with a particular ID to <code>Firefox</code>.


```js
db.collectionName.updateOne(
  { _id: ObjectId("60df129dad74d9467ceebd51") },
  { $set: { "device.Browser": "Firefox" } }
);
```

<p></p>

```js
// {
//   acknowledged: true,
//   insertedId: null,
//   matchedCount: 0,
//   modifiedCount: 0,
//   upsertedCount: 0
// }
```


To maintain the accuracy of further operations, we'll set it back to the original state.


```js
db.collectionName.updateOne(
  { _id: ObjectId("60df129dad74d9467ceebd51") },
  { $set: { "device.Browser": "Chrome" } }
);
```

<p></p>

```js
// {
//   acknowledged: true,
//   insertedId: null,
//   matchedCount: 0,
//   modifiedCount: 0,
//   upsertedCount: 0
// }
```



### Update <code>device.Browser</code> Records to be <code>Firefox</code> if Set to <code>Firefox iOS</code>


If we wanted to update a list of records, we would use <code>updateMany</code> along with <code>$set</code>. Below is an example of how we would change the <code>device.Browser</code> to <code>Firefox</code> where it is currently set to <code>Firefox iOS</code>. For data accuracy, I'll refrain from actually executing the command.


```js
// db.clicks.updateMany(
//   { "device.Browser": "Firefox iOS" },
//   { $set: { "device.Browser": "Firefox" } }
// );
```



### Create New Field


What if we want to create an entirely new field? Well, similar to a new database or collection, there is no need to define it explicitly beforehand. When data is inserted (and only when data gets inserted), then the new dimension is instantiated. 

Below, we'll simply use <code>$set</code> along with the name of our new toy field, and the value we wish to populate. I've limited it to 1000 records because it takes a long time to create a new field for all 6.1M records, although we'll do that in the next article using the better-suited PyMongo.


### Add Field Called <code>NewField</code> to First 1000 Records, Set Value to <code>Default</code>

```js
db.clicks.find().limit(1000).forEach(doc => {
  db.clicks.updateOne(
    { _id: doc._id },
    { $set: { NewField: "Default" } }
  );
});
```

We view a record to confirm the update:

```js
db.clicks.findOne({ NewField: "Default" });
```

<p></p>

```js
// {
//   _id: ObjectId('60df1029ad74d9467c91a932'),
//   webClientID: 'WI100000244987',
//   VisitDateTime: ISODate('2018-05-25T04:51:14.179Z'),
//   ProductID: 'Pr100037',
//   Activity: 'click',
//   device: { Browser: 'Firefox', OS: 'Windows' },
//   user: { City: 'Colombo', Country: 'Sri Lanka' },
//   NewField: 'Default'
// }
```



### Remove the Added Field


Naturally, you're wondering, how do we remove a field? We'll return to our toy example, and fittingly, use the <code>$unset</code> operator to remove it. We can achieve the effect by setting the value to blank, as below.


```js
db.clicks.updateMany(
  { NewField: { $exists: true } },
  { $unset: { NewField: "" } }
);
```

<p></p>

```js
// {
//   acknowledged: true,
//   insertedId: null,
//   matchedCount: 1000,
//   modifiedCount: 1000,
//   upsertedCount: 0
// }
```

<p></p>

It's also the case that we could use the following, not specifying blanks as the new value, but using a <code>1</code> to indicate the field should be applied to the <code>$unset</code> command.


```js
db.clicks.updateMany(
  { [fieldToRemove]: { $exists: true } },
  { $unset: { [fieldToRemove]: 1 } }
);
````




# Indexes

## View Indexes


Indexes are structures that improve the speed and efficiency of queries by creating a sorted mapping from the indexed fields to the location of documents. This allows a query to utilize information about which documents may be ignored, resulting in the search for applicable records to not require a scan of every document in the collection. The <code>mongorestore</code> command does automatically associate indexes listed in the metadata (<code>.json</code> file) with the data (the <code>.bson</code>) file.

To check which indexes exist, we can use the following.

```js
db.clicks.getIndexes()
```

<p></p>

```js
[ { v: 2, key: { _id: 1 }, name: '_id_' } ]
```

<p>We see it only relates to the <code>_id</code> field, which will always be an index by default.</p>


```js
db.clicks.metadata.findOne()
```

<p></p>

```js
// {
//   _id: ObjectId('6837ada071d28360c34516c3'),
//   indexes: [ { v: 2, key: { _id: 1 }, name: '_id_' } ],
//   uuid: 'ee6da5fe5bdf42b2bc3cecee40723af6',
//   collectionName: 'clicks'
```


## Create Indexes

To create indexes, we can do the following. This may come in handy toward the actions taken in the next article, which will involve generating insights from the data, partly based on user device information such as their operating system and browser.

We'll create an index for <code>device.OS</code>.

```js
db.clicks.createIndex({ "device.OS": 1 });
```

<p></p>

```js
// device.OS_1
```


And then <code>device.Browser</code>

```js
db.clicks.createIndex({ "device.Browser": 1 });
```

<p></p>

```js
// device.OS_1
```


# What's Next?

That's it for the basics. Now we can focus on demonstrating the power of unstructured data, through aggregation pipelines for business insights, and machine learning upon text data. We have not touched upon horizontal scaling, but it is certainly the case that one advantage MongoDB has over structured databases is that we can deal with read and write operations at massive scale (e.g., if the clickstream data was streaming in from Amazon). See you at the next article.



# References

Mongo DB User Docs
- <a href="https://www.mongodb.com/docs/">https://www.mongodb.com/docs/</a>




