import re

PREFIX = '../third/'

def extract(filename):
	with open(filename, 'r') as file:
		content = file.read().replace('\n', ' ')
	content = re.split('(Wallace:|Clinton:|Trump:)', content)
	delimiters = {'Wallace:', 'Clinton:', 'Trump:'}
	i = 0
	count = 0
	print ("Extracting quotes...")
	while i < len(content):
		j = i+1
		current = ''
		if content[i] in delimiters:
			while j < len(content) and content[j] not in delimiters:
				current += content[j]
				j += 1
			# write to a new file
			filename = PREFIX + content[i][:-1].lower() + '_' + str(count) + '.txt'
			new_file = open(filename, "w")
			new_file.write(current)
			new_file.close()
			count += 1
		i = j
	return

if __name__ == "__main__":
	temp = extract('full2.txt')
