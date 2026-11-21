# Programowanie w CUDA na NVIDIA GPU (PKG) – sprawozdanie z projektu

**Projekt:** Suma elementów kwadratowego okna (2R+1)×(2R+1) tablicy N×N na GPU (CUDA)

**Autorzy:**
- Mateusz Chlebosz, 151817
- TODO (imię nazwisko, nr indeksu – drugi członek grupy, jeśli dołączy)
**Data oddania:** TODO

---

## 1. Opis systemów użytych w eksperymencie

### 1.1. Procesory (CPU)
<!-- model, liczba rdzeni (typy rdzeni), max wątków, moc nominalna + źródło -->
TODO

### 1.2. Karty graficzne (GPU)
<!-- model, układ, moc znamionowa, architektura, CC, liczba SM, pamięci, ograniczenia CC (shared/blok, wątki/SM, banki, rejestry) -->
TODO

## 2. Zakres zrealizowanego zadania
TODO

## 3. Realizacja projektu

### 3.1. Program referencyjny CPU (OpenMP)
TODO

### 3.2. Konfiguracja uruchomienia kernela
TODO

### 3.3. Wariant 1 – dostęp efektywny (pamięć globalna łączona, shared bez konfliktów)
TODO – listing, zmienne prywatne/współdzielone, lokalność, rysunki dostępu

### 3.4. Wariant 2 – dostęp nieefektywny (pamięć globalna niełączona, shared z konfliktami)
TODO – listing, rysunki dostępu

### 3.5. Test poprawności
TODO

### 3.6. Wzory: wydajność, energia, CGMA
TODO

## 4. Wyniki pomiarów

### 4.1. Wyznaczenie N_stab
TODO – tabela dla R1, tabela dla R2

### 4.2. Porównanie CPU i wariantów GPU
TODO

## 5. Wnioski
TODO
