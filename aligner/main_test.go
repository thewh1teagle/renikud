package main

import (
	"encoding/csv"
	"os"
	"strings"
	"testing"
)

var ipaNormalize = map[string]string{
	"sh": "ʃ",
	"x":  "χ",
	"g":  "ɡ",
	"r":  "ʁ",
}

func normalizeIPA(s string) string {
	for ascii, ipa := range ipaNormalize {
		s = strings.ReplaceAll(s, ascii, ipa)
	}
	return s
}

func hebrewOnly(word string) string {
	var b strings.Builder
	for _, r := range word {
		if r >= 'א' && r <= 'ת' {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func loadCSV(path string) [][]string {
	f, _ := os.Open(path)
	defer f.Close()
	rows, _ := csv.NewReader(f).ReadAll()
	return rows[1:] // skip header
}

func TestAlignWordValid(t *testing.T) {
	for _, file := range []string{"basic.csv", "advanced.csv"} {
		for _, row := range loadCSV("testdata/" + file) {
			heb, alignment := row[0], row[1]
			spans := strings.Split(alignment, "|")
			ipa := strings.Join(func() []string {
				clean := make([]string, len(spans))
				for i, s := range spans {
					clean[i] = strings.ReplaceAll(normalizeIPA(s), "∅", "")
				}
				return clean
			}(), "")
			expected := make([]string, len(spans))
			for i, s := range spans {
				expected[i] = strings.ReplaceAll(normalizeIPA(s), "∅", "")
			}

			result := alignWord(hebrewOnly(heb), ipa)
			if result == nil {
				t.Errorf("[%s] %q: alignWord returned nil for ipa %q", file, heb, ipa)
				continue
			}
			if len(result) != len(expected) {
				t.Errorf("[%s] %q: got %d chunks, want %d", file, heb, len(result), len(expected))
				continue
			}
			for i, pair := range result {
				if pair[1] != expected[i] {
					t.Errorf("[%s] %q: chunk %d: got %q, want %q", file, heb, i, pair[1], expected[i])
				}
			}
		}
	}
}

func TestAlignWordInvalid(t *testing.T) {
	for _, row := range loadCSV("testdata/invalid.csv") {
		heb, ipa := row[0], row[1]
		if result := alignWord(hebrewOnly(heb), ipa); result != nil {
			t.Errorf("%q + %q: expected nil, got %v", heb, ipa, result)
		}
	}
}
