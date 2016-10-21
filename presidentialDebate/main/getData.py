import re

# TODO: pass as command line argument
FILES = ['./data/full0.txt', './data/full1.txt', './data/full2.txt']
PREFIXES = ['./data/first/', './data/second/', './data/third/']
SPLITS = ['(Holt:|Clinton:|Trump:)', '(Raddatz:|Cooper:|Clinton:|Trump:)', '(Wallace:|Clinton:|Trump:)']
DELIMITERS = [{'Holt:', 'Clinton:', 'Trump:'}, {'Raddatz:', 'Cooper:', 'Clinton:', 'Trump:'}, {'Wallace:', 'Clinton:', 'Trump:'}]

def extract(filename, n):
	with open(filename, 'r') as file:
		content = file.read().replace('\n', ' ')
	content = re.split(SPLITS[n], content)
	delimiter = DELIMITERS[n]
	i = 0
	count = 0
	print ("Extracting quotes from %s" % filename)
	while i < len(content):
		j = i+1
		current = ''
		if content[i] in delimiter:
			while j < len(content) and content[j] not in delimiter:
				current += content[j]
				j += 1
			# write to a new file
			filename = PREFIXES[n] + content[i][:-1].lower() + '_' + str(count) + '.txt'
			new_file = open(filename, "w")
			new_file.write(current)
			new_file.close()
			count += 1
		i = j
	return

if __name__ == "__main__":
	count = 0
	for file in FILES:
		extract(file, count)
		count += 1