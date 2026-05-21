# ¡Bienvenido a 3-Adic ML! 🌌

Bienvenido al proyecto **3-Adic ML**. Estamos explorando una intersección fascinante entre las matemáticas puras, la geometría y el aprendizaje profundo (deep learning). Si eres nuevo aquí, esta guía te ayudará a entender qué estamos haciendo y por qué es importante.

## 🧠 La Gran Idea: "Significado = Geometría"

La mayoría de los modelos de IA aprenden patrones de los datos, pero a menudo tienen dificultades con la **jerarquía**. Por ejemplo, un "Golden Retriever" es un tipo de "Perro", que es un "Mamífero", que es un "Animal".

En este proyecto, creemos que **la jerarquía debe estar integrada en la forma del espacio donde la IA "piensa"**.

### 1. Números P-ádicos (La Lógica)
Utilizamos la **valuación 3-ádica** como nuestra base matemática. Es una forma de medir la "divisibilidad". En nuestro sistema, algunos números son "más fundamentales" que otros según cuántas veces pueden ser divididos por 3. Esto nos da una estructura de árbol matemática natural.

### 2. Geometría Hiperbólica (El Mapa)
Imagina un mapa donde cuanto más te alejas del centro, más "espacio" hay. Eso es una **Bola de Poincaré** (un modelo de geometría hiperbólica). Es el hogar perfecto para estructuras en forma de árbol porque puede albergar un número infinito de ramas sin que se amontonen.

### 3. El Resultado
Entrenamos a una IA (un Autoencoder Variacional o VAE) para mapear datos en este espacio hiperbólico.
- **Las cosas fundamentales** (valuación alta) se sitúan cerca del **centro**.
- **Los detalles específicos** (valuación baja) se sitúan cerca de los **bordes**.
- **Las cosas similares** se sitúan en la misma **dirección**.

La jerarquía no es solo algo que se aprende; es algo que **emerge de la geometría**.

## 🚀 ¿Por qué estamos haciendo esto?

*   **Mejor Razonamiento de la IA:** Los modelos que entienden la jerarquía de forma nativa pueden razonar mejor sobre estructuras complejas como el código, la biología o el lenguaje.
*   **Descubrimiento Científico:** Estamos aplicando esto a la bioinformática para entender cómo se organizan las proteínas y los genes.
*   **Eficiencia:** Podemos representar relaciones complejas en muy pocas dimensiones.

## 🛠️ Cómo Explorar

1.  **Lee el [README](../README.md):** Para la configuración técnica y la arquitectura.
2.  **Consulta las Visualizaciones:** Durante el entrenamiento, generamos mapas interactivos en 3D (archivos HTML) que muestran cómo la IA organiza los datos.
3.  **Explora las [Preguntas Frecuentes (FAQ)](FAQ.md):** Para respuestas a preguntas comunes.
4.  **Únete a la Investigación:** Mira nuestro [STATUS.md](STATUS.md) para ver en qué estamos trabajando actualmente (Fase 10+).

## 🌍 Comunidad

Somos un grupo de investigación pequeño y enfocado. Valoramos la integridad técnica y la belleza matemática. Ya seas un entusiasta de las matemáticas, un ingeniero de aprendizaje profundo o simplemente alguien curioso, ¡nos alegra que estés aquí!

## 🔢 ¿Qué son los números 3-ádicos?

Para entender nuestro proyecto, no necesitas ser un experto en matemáticas, solo imaginar los números de una forma diferente.

Normalmente, pensamos que dos números están "cerca" si su diferencia es pequeña (como 1 y 2). En el mundo **p-ádico** (donde $p=3$ en nuestro caso), la cercanía se mide por la **divisibilidad**.
- El número **3** es más "pequeño" o "fundamental" que el 1.
- El número **9** ($3^2$) es aún más fundamental.
- El número **27** ($3^3$) lo es todavía más.

Esto crea una **valuación**. Cuanto más veces sea un número divisible por 3, más cerca está del "centro" de nuestro sistema. Matemáticamente, esto genera un **espacio ultramétrico**: una estructura que se comporta como un árbol infinito donde cada rama se divide en tres.

## 🧬 La Conexión con la Biología: Nucleótidos y Codones

¿Por qué usar matemáticas de base 3 para la vida? La biología es jerárquica por naturaleza:

1.  **Codones y Tripletes**: La vida se escribe en ramilletes de tres. Un **codón** es una secuencia de 3 nucleótidos que codifica un aminoácido. Esta estructura de "triplete" resuena perfectamente con la lógica ternaria (base 3) de nuestro sistema.
2.  **La Jerarquía de la Evolución**: El ADN no es solo una cadena de letras; es un registro histórico. Las mutaciones que ocurren en posiciones críticas son más "fundamentales" que otras. La valuación 3-ádica nos permite mapear estas prioridades:
    *   **Nivel 0**: Cambios superficiales.
    *   **Niveles Superiores**: Cambios estructurales en proteínas o funciones vitales.
3.  **Geometría del Genoma**: Al mapear nucleótidos (A, C, G, T) a coordenadas 3-ádicas, podemos visualizar el genoma no como un texto plano, sino como un **paisaje geométrico** en una Bola de Poincaré.

## 🌡️ ¿Cómo puede cambiar la Bioinformática?

La aplicación de los VAEs 3-ádicos a la bioinformática (nuestro proyecto hermano **[3-Adic Bioinformatics](https://github.com/gesttaltt/3-adic-bioinformatics)**) busca revolucionar el área de tres formas:

*   **Compresión Semántica**: En lugar de guardar gigabytes de secuencias "muertas", guardamos la **geometría del significado**. Podemos comprimir genomas enteros preservando las relaciones jerárquicas entre genes.
*   **Predicción de Plegamiento de Proteínas**: Las proteínas se pliegan siguiendo jerarquías de energía. La geometría hiperbólica es ideal para modelar cómo las subestructuras pequeñas se combinan en estructuras grandes sin "colisionar" en el espacio de cómputo.
*   **Búsqueda Ultra-rápida**: En un espacio ultramétrico, encontrar secuencias similares es órdenes de magnitud más rápido. No comparamos letra por letra; comparamos "ramas del árbol" en el espacio hiperbólico.

---

*"La vida no lee texto, la vida habita una geometría. Estamos aquí para descifrarla."*
