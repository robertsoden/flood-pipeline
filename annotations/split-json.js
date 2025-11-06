#!/usr/bin/env node

/**
 * JSON Random Splitter
 * 
 * Removes a random selection of items from a large JSON file and saves them
 * to a new file, updating the original file with remaining items.
 * 
 * Usage:
 *   node split-json.js <input-file> <count>
 * 
 * Example:
 *   node split-json.js data.json 100
 */

const fs = require('fs');
const path = require('path');

function shuffleArray(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
}

function splitJsonFile(inputFile, count) {
    // Validate input file exists
    if (!fs.existsSync(inputFile)) {
        console.error(`Error: Input file '${inputFile}' not found`);
        process.exit(1);
    }

    // Read and parse the input file
    console.log(`Reading ${inputFile}...`);
    const data = JSON.parse(fs.readFileSync(inputFile, 'utf8'));

    // Ensure data is an array
    if (!Array.isArray(data)) {
        console.error('Error: Input file must contain a JSON array');
        process.exit(1);
    }

    // Validate count
    if (count <= 0) {
        console.error('Error: Count must be a positive number');
        process.exit(1);
    }

    if (count >= data.length) {
        console.error(`Error: Cannot remove ${count} items from array of length ${data.length}`);
        process.exit(1);
    }

    // Shuffle and split
    console.log(`Total items: ${data.length}`);
    console.log(`Randomly selecting ${count} items...`);
    
    const shuffled = shuffleArray(data);
    const removed = shuffled.slice(0, count);
    const remaining = shuffled.slice(count);

    // Generate output filename
    const ext = path.extname(inputFile);
    const base = path.basename(inputFile, ext);
    const dir = path.dirname(inputFile);
    const outputFile = path.join(dir, `${base}_subset_${count}${ext}`);

    // Create backup of original file
    const backupFile = inputFile + '.backup';
    console.log(`Creating backup: ${backupFile}`);
    fs.copyFileSync(inputFile, backupFile);

    // Write files
    console.log(`Writing ${count} items to ${outputFile}...`);
    fs.writeFileSync(outputFile, JSON.stringify(removed, null, 2));

    console.log(`Writing ${remaining.length} remaining items back to ${inputFile}...`);
    fs.writeFileSync(inputFile, JSON.stringify(remaining, null, 2));

    console.log('\nDone!');
    console.log(`Original file (${inputFile}): ${data.length} → ${remaining.length} items`);
    console.log(`Subset file (${outputFile}): ${removed.length} items`);
    console.log(`Backup saved as: ${backupFile}`);
}

// Parse command line arguments
const args = process.argv.slice(2);

if (args.length !== 2) {
    console.log('Usage: node split-json.js <input-file> <count>');
    console.log('\nExample:');
    console.log('  node split-json.js data.json 100');
    console.log('\nThis will create data_subset_100.json with 100 random items');
    process.exit(1);
}

const inputFile = args[0];
const count = parseInt(args[1], 10);

if (isNaN(count)) {
    console.error('Error: Count must be a valid number');
    process.exit(1);
}

splitJsonFile(inputFile, count);
